# TensorFlow-WESPE (Ongoing as of 2018.10.2)
TensorFlow implementation of "WESPE: Weakly Supervised Photo Enhancer for Digital Cameras" in CVPRW 2018 [1]

## **How to run the code**
- In WESPE-main.ipynb (plan to add main.py soon)
- Model & training code for the previous model, DPED (in ICCV 2017) [2] can be found in DPED.py & DPED-main.ipynb (full version of my implemented DPED can be found in https://github.com/JuheonYi/DPED-Tensorflow/)

## **TODO list**
- [x] Implement WESPE model
- [x] Train WESPE under strong supervision with DPED dataset (WESPE[DPED])
- [x] Implement weakly supervised dataloader for DPED & DIV2K
- [ ] Train WESPE under weak supervision (WESPE[DIV2K]) --> ongoing

## **Training result**
1. Training log 
   - Strong supervision: WESPE[DPED]-main.ipynb (test PSNR for iphone is about 17.5 dB, similar to the original paper, 18.11 dB)
   - Weak supervision: WESPE[DIV2K]-main.ipynb
2. Trained model
   - In "./result/" directory 
3. Visual result (to be updated soon...) 
   - WESPE[DPED]
![Example result](https://github.com/JuheonYi/images/blob/master/WESPE_strong.PNG)
   - WESPE[DIV2K]

## **Modifications**
1. Model architecture
   - Removed batch normalization layer in generator
   - Replaced the tanh layer to 1x1 convolution 
   - (refer to https://github.com/JuheonYi/DPED-Tensorflow/ for reasons)
2. WESPE[DPED] (strong supervision)
   - I used the 'relu_2_2' layer for content loss (using the ealry layers in VGG19 leads to better reconstruction performance)
   - Removed blurring for computing color loss
   - Weights for each losses are configured as (w_content, w_color, w_texture, w_tv) = (0.1, 25, 3, 1/400). w_content should be small enough so that the generator focus more on enhancement rather than reconstruction. w_color seems to be the most important. w_texture also plays an important role in reducing unwanted artifacts in the generated image.
   - Training seems to suffer from mode collapse (when trained wrong, enhanced image looks like negatively processed image). When Enhanced-Original PSNR seems to stay below 10 dB during training, stopped the training and started from beginning again. 
3. WESPE[DIV2K] (weak supervision)
   - In process of finding the right weight combinations...


## **References**
- [1] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey and Luc Van Gool. "WESPE: Weakly Supervised Photo Enhancer for Digital Cameras," in IEEE International Conference on Computer Vision and Pattern Recognition Workshop (CVPRW), 2018.
- [2] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey and Luc Van Gool. "DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks," in IEEE International Conference on Computer Vision (ICCV), 2017.

