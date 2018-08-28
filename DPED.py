from __future__ import division
import os
import time
import tensorflow as tf
import scipy.misc
import scipy.io
import numpy as np
from glob import glob
from utils import *
from ops import *
from vgg19 import *
from dataloader import *
import modules

class DPED(object):
    def __init__(self, sess, config, dataset_phone, dataset_dslr):
        # copy training parameters
        self.sess = sess
        self.config = config
        self.batch_size = config.batch_size
        self.patch_size = config.patch_size
        self.mode = config.mode
        self.channels = config.channels
        self.augmentation = config.augmentation
        self.checkpoint_dir = config.checkpoint_dir
        
        self.content_layer = config.content_layer
        self.vgg_dir = config.vgg_dir
        
        self.dataset_name = config.dataset_name
        self.dataset_phone = dataset_phone
        self.dataset_dslr = dataset_dslr
        
        # loss weights
        self.w_content = config.w_content
        self.w_texture = config.w_texture 
        self.w_color = config.w_color
        self.w_tv = config.w_tv
        
        # patches for training (fixed size)
        self.phone_patch = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.channels], name='input_phone_patch') 
        self.dslr_patch = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.channels], name='input_dslr_patch') 
        
        # images for testing (unknown size)
        self.phone_test = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.channels], name='input_phone_test')
        self.phone_test_unknown = tf.placeholder(tf.float32, [None, None, None, self.channels], name='input_phone_test_unknown_size')
        self.dslr_test = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.channels], name='input_dslr_test')
        
        # input to discriminator network
        self.input_discriminator = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.channels], name='input_discriminator') 
        
        # builc models
        self.build_generator()
        self.build_discriminator()
        
        # build loss function (color + texture + content + TV)
        self.build_generator_loss()
        tf.global_variables_initializer().run(session=self.sess)
        
        self.saver = tf.train.Saver(tf.trainable_variables())

    def build_generator(self):
        self.enhanced_patch = modules.generator_network(self.phone_patch, var_scope = 'generator')
        self.enhanced_test = modules.generator_network(self.phone_test, var_scope = 'generator')
        self.enhanced_test_unknown = modules.generator_network(self.phone_test_unknown, var_scope = 'generator')
        
        variables = tf.trainable_variables()
        self.g_var = [x for x in variables if 'generator' in x.name]
        print("Completed building generator. Number of variables:",len(self.g_var))
        #print(self.g_var)

    def build_generator_loss(self):
        # color loss (blur + mse) - since output values are normalized, color loss should be multiplied by 255
        self.color_loss = 255 * tf.reduce_mean(tf.square(gaussian_blur(self.dslr_patch)-gaussian_blur(self.enhanced_patch)))
        
        # texture loss (gan)
        self.texture_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_dslr, self.logits_enhanced))
        
        # content loss (vgg)
        enhanced_vgg = net(self.vgg_dir, self.enhanced_patch * 255)
        dslr_vgg = net(self.vgg_dir, self.dslr_patch * 255)
        self.content_loss = tf.reduce_mean(tf.square(enhanced_vgg[self.content_layer] - dslr_vgg[self.content_layer])) 
        
        # tv loss (tv)
        self.tv_loss = tf.reduce_mean(tf.image.total_variation(self.enhanced_patch))
        
        # calculate generator loss as a weighted sum of the above 4 losses
        self.G_loss = self.color_loss * self.w_color + self.texture_loss * self.w_texture + self.content_loss * self.w_content + self.tv_loss * self.w_tv
        self.G_optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.G_loss, var_list=self.g_var)
    
    def build_discriminator(self): 
        self.logits_phone, _ = modules.discriminator_network(self.phone_patch, var_scope = 'discriminator')
        self.logits_dslr, _ = modules.discriminator_network(self.dslr_patch, var_scope = 'discriminator')
        self.logits_enhanced, _ = modules.discriminator_network(self.enhanced_patch, var_scope = 'discriminator')
        
        _, self.prob = modules.discriminator_network(self.phone_test)
           
        variables = tf.trainable_variables()
        self.d_var = [x for x in variables if 'discriminator' in x.name]
        print("Completed building discriminator. Number of variables:",len(self.d_var))
        
        d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_dslr, tf.ones_like(self.logits_dslr)))
        d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_phone, tf.zeros_like(self.logits_phone)))
        
        self.d_loss = d_loss_real + d_loss_fake
        self.D_optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss, var_list=self.d_var)
    
    def pretrain_discriminator(self, load = True):
        if load == True:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" Discriminator training starts from beginning")
        start = time.time()
        for i in range(0, 10000):
            phone_batch, dslr_batch = get_batch(self.dataset_phone, self.dataset_dslr, self.config)
            _ = self.sess.run(self.D_optimizer , feed_dict={self.phone_patch:phone_batch, self.dslr_patch:dslr_batch})
            
            if i %2000 == 0:
                phone_batch, dslr_batch = get_batch(self.dataset_phone, self.dataset_dslr, self.config)
                d_loss = self.sess.run(self.d_loss , feed_dict={self.phone_patch:phone_batch, self.dslr_patch:dslr_batch})
                print("Iteration %d, runtime: %.3f s, discriminator loss: %.6f" %(i, time.time()-start, d_loss))
                self.test_discriminator(200)
        print("pretraining complete")
        self.save()
    
    def train(self, load = True):
        if load == True:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" Overall training starts from beginning")
        start = time.time()
        for i in range(0, 100000):
            phone_batch, dslr_batch = get_batch(self.dataset_phone, self.dataset_dslr, self.config)
            _, enhanced_batch = self.sess.run([self.G_optimizer, self.enhanced_patch] , feed_dict={self.phone_patch:phone_batch, self.dslr_patch:dslr_batch})
            _ = self.sess.run(self.D_optimizer , feed_dict={self.phone_patch:enhanced_batch, self.dslr_patch:dslr_batch})
            
            if i %1000 == 0:
                phone_batch, dslr_batch = get_batch(self.dataset_phone, self.dataset_dslr, self.config)
                #g_loss = self.sess.run(self.G_loss , feed_dict={self.phone_patch:phone_batch, self.dslr_patch:dslr_batch})
                g_loss, color_loss, texture_loss, content_loss, tv_loss = self.sess.run([self.G_loss, self.color_loss, self.texture_loss, self.content_loss, self.tv_loss] , feed_dict={self.phone_patch:phone_batch, self.dslr_patch:dslr_batch})
                print("Iteration %d, runtime: %.3f s, generator loss: %.6f" %(i, time.time()-start, g_loss))
                print("Loss per component: color %.6f, texture %.6f, content %.6f, tv %.6f" %(color_loss, texture_loss, content_loss, tv_loss)) 
                # during training, test for only patches (full image testing incurs memory issues...)
                self.test_generator(200, 0)
                self.save()
  
    def test_discriminator(self, test_num, load = False, mode = "phone_dslr"):
        if load == True:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        #print("testing discriminator")
        test_list_dslr = sorted(glob(self.config.test_path_dslr_patch))
        test_list_phone = sorted(glob(self.config.test_path_phone_patch))
        #print("total testset: %d image pairs" %len(test_list_dslr))
        # test dlsr
        acc_dslr = 0
        acc_phone = 0
        acc_enhanced = 0
        indexes = []
        probs = np.zeros([test_num])
        for i in range(test_num):
            index = np.random.randint(len(test_list_dslr))
            indexes.append(index)
            test_patch_dslr = preprocess(scipy.misc.imread(test_list_dslr[index], mode = "RGB").astype("float32"))
            prob = self.sess.run(self.prob, feed_dict={self.phone_test: [test_patch_dslr]})
            if prob > 0.5:
                probs[i] = prob
                acc_dslr += 1
            test_patch_phone = preprocess(scipy.misc.imread(test_list_phone[index], mode = "RGB").astype("float32"))
            prob = self.sess.run(self.prob, feed_dict={self.phone_test: [test_patch_phone]})
            if prob < 0.5:
                acc_phone += 1
            if mode == "enhanced":
                test_patch_enhanced = self.sess.run(self.enhanced_test , feed_dict={self.phone_test:[test_patch_phone], self.dslr_test:[test_patch_dslr]})
                prob = self.sess.run(self.prob, feed_dict={self.phone_test: [test_patch_enhanced[0]]})
                if prob < 0.5:
                    acc_enhanced += 1
        if mode == "enhanced":
            print("Dricriminator test accuracy: phone: %d/%d, dslr: %d/%d, enhanced: %d/%d" %(acc_phone, test_num, acc_dslr, test_num, acc_enhanced , test_num)) 
        else:
            print("Discriminator test accuracy: phone: %d/%d, dslr: %d/%d" %(acc_phone, test_num, acc_dslr, test_num))    
        
    def test_generator(self, test_num_patch = 200, test_num_image = 5, load = False):
        if load == True:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # test for patches
        start = time.time()
        self.test_discriminator(200, load = False, mode = "enhanced")
        test_list_phone = sorted(glob(self.config.test_path_phone_patch))
        test_list_dslr = sorted(glob(self.config.test_path_dslr_patch))
        PSNR_phone_enhanced_list = np.zeros([test_num_patch])
        PSNR_dslr_enhanced_list = np.zeros([test_num_patch])
        indexes = []
        for i in range(test_num_patch):
            index = np.random.randint(len(test_list_dslr))
            indexes.append(index)
            test_patch_phone = preprocess(scipy.misc.imread(test_list_phone[index], mode = "RGB").astype("float32"))
            test_patch_dslr = preprocess(scipy.misc.imread(test_list_dslr[index], mode = "RGB").astype("float32"))
            test_patch_enhanced = self.sess.run(self.enhanced_test , feed_dict={self.phone_test:[test_patch_phone], self.dslr_test:[test_patch_dslr]})
            if i % 50 == 0:
                imageio.imwrite(("./samples/%s/patch/phone_%d.png" %(self.config.dataset_name, i)), postprocess(test_patch_phone))
                imageio.imwrite(("./samples/%s/patch/dslr_%d.png" %(self.config.dataset_name,i)), postprocess(test_patch_dslr))
                imageio.imwrite(("./samples/%s/patch/enhanced_%d.png" %(self.config.dataset_name,i)), postprocess(test_patch_enhanced[0]))
            #print(enhanced_test_patch.shape)
            PSNR = calc_PSNR(postprocess(test_patch_enhanced[0]), postprocess(test_patch_phone))
            #print("PSNR: %.3f" %PSNR)
            PSNR_phone_enhanced_list[i] = PSNR
            PSNR = calc_PSNR(postprocess(test_patch_enhanced[0]), postprocess(test_patch_dslr))
            #print("PSNR: %.3f" %PSNR)
            PSNR_dslr_enhanced_list[i] = PSNR
        print("(runtime: %.3f s) Average test PSNR for %d random test image patches: phone-enhanced %.3f, dslr-enhanced %.3f" %(time.time()-start, test_num_patch, np.mean(PSNR_phone_enhanced_list), np.mean(PSNR_dslr_enhanced_list) ))
        
        # test for images
        start = time.time()
        test_list_phone = sorted(glob(self.config.test_path_phone_image))
        PSNR_phone_enhanced_list = np.zeros([test_num_image])
        PSNR_dslr_enhanced_list = np.zeros([test_num_image])
        indexes = []
        for i in range(test_num_image):
            #index = np.random.randint(len(test_list_phone))
            index = i
            indexes.append(index)
            test_image_phone = preprocess(scipy.misc.imread(test_list_phone[index], mode = "RGB").astype("float32"))
            test_image_enhanced = self.sess.run(self.enhanced_test_unknown , feed_dict={self.phone_test_unknown:[test_image_phone]})
            imageio.imwrite(("./samples/%s/image/phone_%d.png" %(self.config.dataset_name, i)), postprocess(test_image_phone))
            imageio.imwrite(("./samples/%s/image/enhanced_%d.png" %(self.config.dataset_name, i)), postprocess(test_image_enhanced[0]))
            PSNR = calc_PSNR(postprocess(test_image_enhanced[0]), postprocess(test_image_phone))
            #print("PSNR: %.3f" %PSNR)
            PSNR_phone_enhanced_list[i] = PSNR
        if test_num_image > 0:
            print("(runtime: %.3f s) Average test PSNR for %d random full test images: phone-enhanced %.3f" %(time.time()-start, test_num_image, np.mean(PSNR_phone_enhanced_list)))

    def save(self):
        model_name = self.config.model_name
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.dataset_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), write_meta_graph=False)

    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.dataset_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("Loading checkpoints from ",checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            model_name = self.config.model_name
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, model_name))
            return True
        else:
            return False