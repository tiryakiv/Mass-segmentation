# Mass-segmentation and classification from film mammograms

This repository includes mass segmentation investigation using U-net++Xception-based segmentation models. The first step includes a breast segmentation and the second step includes the mass segmentation. The two step enabled to focus only on the mass segmentation.

In the first step five-layer U-net was found to have the hightest breast segmentation performance.

In the second step U-net++Xception had the highest mass segmentation performance. The proposed U-net++Xception model has statistically siginificant better performance than U-net5L, Unet++, ResUnet, DeepLabV3Plus and AttentionU-net in terms of DSC. 

The references for models are given in the top line as a comment.

## Software configuration:
* Windows 10 Pro - 64 bit
* Miniconda3 Python 3.9
* cuda_11.8.0_522.06_windows
* Tensorflow 2.10
* Keras 2.6
* cudatoolkit 11.2.2
* cudnn 8.1.0.77
