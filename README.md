# TensorFlow-WESPE (Ongoing as of 2018.8.28)
TensorFlow implementation of WESPE [1]

## **How to run the code**
In WESPE-main.ipynb (plan to add main.py soon)
Model & training code for the previous version, DPED (in ICCV 2017) [3] is also included in DPED.py & DPED-main.ipynb (Codes for DPED can also be found on [4])

## **TODO list**
- [x] Implement WESPE model
- [] Train WESPE under strong supervision with DPED dataset (WESPE 'DPED') -> ongoing
- [] Implement weakly supervised dataloader for DPED & DIV2K
- [] Train WESPE under weak supervision (WESPE 'DIV2K')
- [] Implement Flickr Faves Score estimation network
- [] Final evaluation (both qualitative & quantatative) on various datasets (DPED, KITTI, Cityscapes, and various smartphones)

## **Training result**
1. Training log 
   - In WESPE-main.ipynb 
2. Trained model
   - In "./result/" directory 
3. Visual result (to be updated soon...) 

## **References**
- [1] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey and Luc Van Gool. "WESPE: Weakly Supervised Photo Enhancer for Digital Cameras," in IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- [2] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, “Enhanced Deep Residual Networks for Single Image Super-Resolution,” in IEEE CVPR Workshops, 2017.
- [3] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey and Luc Van Gool. "DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks," in IEEE International Conference on Computer Vision (ICCV), 2017.
- [4] https://github.com/JuheonYi/DPED-Tensorflow/

