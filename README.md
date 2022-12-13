# Mass-segmentation and classification from film mammograms

This repository includes mass segmentation investigation using U-net++Xception-based segmentation models. The first step includes a breast segmentation and the second step includes the mass segmentation. The two step enabled to focus only on the mass segmentation.

In the first step five-layer U-net was found to have the hightest breast segmentation performance.

In the second step U-net++Xception had the highest mass segmentation performance. The proposed U-net++Xception model has statistically siginificant better performance than U-net5L, Unet++, ResUnet, DeepLabV3Plus and AttentionU-net in terms of DSC. 

Breast segmentation training and validation codes:


Mass segmentation training and validation codes:
* train_valid_ResUnet_mass_seg08_14nov22_v011.ipynb
* train_valid_basic_AUnet_mass_seg08_18nov22_v017.ipynb
* train_valid_deepLabV3plus_mass_seg08_17nov22_v015.ipynb
* train_valid_multiResUnet_mass_seg08_17nov22_v016.ipynb
* train_valid_uNetPlusPlusXcept_mass_seg08_16nov22_v013.ipynb
* train_valid_uResNet50_mass_seg08_12nov22_v009.ipynb
* train_valid_uVGG16_mass_seg08_10nov22_v007.ipynb
* train_valid_uXception_mass_seg08_9nov22_v006.ipynb
* train_valid_unet5L_mass_seg08_7nov22_v001.ipynb

Mass segmentation test code:
* test_uNetPlusPlusXcept_mass_seg08_23nov22_v013.ipynb

Mass classification train-valid-test codes:



The references for models are given in the top line as a comment.

## Software configuration:
* Windows 10 Pro - 64 bit
* Python 3.9 - 64 bit (Miniconda3)
* Tensorflow 2.10
* Keras 2.6
* cudatoolkit 11.2.2
* cudnn 8.1.0.77
