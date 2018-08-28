# TensorFlow-WESPE
TensorFlow implementation of WESPE [1]

## **How to run the code**
In WESPE-main.ipynb 

## **Modifications** 

I had some difficulties reproducing the result when training with the same network architecture & parameters in the original paper, so I modified some points as follows.

1. Removed batch normalization layer in generator network, as suggested in [2].
2. Replaced the last layer of the generator network (tanh) to 1x1 convolutional layer (tanh layer didn't seem to train well).
3. Modified weighting factors of each losses. When computing the losses as in build_generator_loss() function in DPED.py (line 104-121), they yielded values in different scales (e.g., color loss in 1e-1 scale, TV loss in 1e2 scale). I modified the weights so that they are scaled to comparable range. My preference was to give highest weight for the content loss (computed with VGG features), as the main difference between smartphone & DSLR pictures seemed to lie in the brightness. While I thought that color loss also was important, I did not give its weight as high as the content loss, as there are some pixel misalignments in the training dataset, as mentioned in the paper. 
4. Added data augmentation in dataloader by flipping, rotating (seems to be helpful since loss functions are shift-variant).
5. Modified learning rate from 5e-4 to 1e-4.


## **Training result**
1. Training log 
   - In WESPE-main.ipynb (PSNR measurements on random test patches yield about 20~21 dB, similar to the original paper)
2. Trained model
   - In "./result/" directory (currently I saved only models for "iphone" and "sony")
3. Visual result (more examples in "./samples/iphone/image/" directory)

![Example result](https://github.com/JuheonYi/DPED-Tensorflow/blob/master/example.PNG)

## **Codes I referenced for implementation**
1. Overall code structure 
   - My own codes [3] for DPED [4], the authors' previous paper in ICCV 2017. 


## **References**
- [1] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey and Luc Van Gool. "WESPE: Weakly Supervised Photo Enhancer for Digital Cameras," in IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- [2] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, “Enhanced Deep Residual Networks for Single Image Super-Resolution,” in IEEE CVPR Workshops, 2017.
- [3] https://github.com/JuheonYi/DPED-Tensorflow/
- [4] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey and Luc Van Gool. "DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks," in IEEE International Conference on Computer Vision (ICCV), 2017.

