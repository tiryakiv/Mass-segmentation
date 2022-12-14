# Mass-segmentation and classification from film mammograms

This repository includes mass segmentation investigation using U-net++Xception-based segmentation models. The first step includes a breast segmentation and the second step includes the mass segmentation. The two step enabled to focus only on the mass segmentation.

In the first step, the five-layer U-net was found to have the hightest breast segmentation performance.

In the second step, the U-net++Xception had the highest mass segmentation performance. The proposed U-net++Xception model has better performance than U-net5L, Unet++, ResUnet, DeepLabV3Plus and AttentionU-net in terms of DSC. 

Breast segmentation training and validation codes:
* train_valid_unet5L_mg_seg04_4nov22_v001.ipynb

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
* mass_class_inceptionv3_xval_28nov22.ipynb
* mass_class_resnet50v2_xval_28nov22.ipynb
* mass_class_vgg16_xval_28nov22.ipynb

Data code names start with data and followed by channel number. Mass segmentation data file is set for 640x640 mammogram size. Breast segmentation data file is set for 1024x768 size.

Deep learning model code names start with "model" and followed by the model name. The references for models are given in the top line as a comment. Deep learning model code names include the date of experiment and followed by version number. Version number is required to distinguish between models. Batch size is two due to the memory limit and to establish fairness for comparing models' performance.

## Training, validation, and testing folder hierarchy for mass segmentation five-fold cross-validation:
```bash
├── mass_seg_08
│   ├── test
│   │   ├── mg
│   │   ├── mask
│   │   ├── pred
│   ├── train01
│   │   ├── mg
│   │   ├── mask
│   ├── train02
│   │   ├── mg
│   │   ├── mask
│   ├── train03
│   │   ├── mg
│   │   ├── mask
│   ├── train04
│   │   ├── mg
│   │   ├── mask
│   ├── train05
│   │   ├── mg
│   │   ├── mask
│   ├── valid01
│   │   ├── mg
│   │   ├── mask
│   │   ├── pred
│   ├── valid02
│   │   ├── mg
│   │   ├── mask
│   │   ├── pred
│   ├── valid03
│   │   ├── mg
│   │   ├── mask
│   │   ├── pred
│   ├── valid04
│   │   ├── mg
│   │   ├── mask
│   │   ├── pred
│   ├── valid05
│   │   ├── mg
│   │   ├── mask
│   │   ├── pred
```


## Software configuration:
* Windows 10 Pro - 64 bit
* Python 3.9 - 64 bit (Miniconda3 package)
* Tensorflow 2.10
* Keras 2.6
* cudatoolkit 11.2.2
* cudnn 8.1.0.77
* ipykernel

## Hardware configuration:
* Dell WS T7610
* Intel Xeon E5-2630 2.6 GHz CPU
* GeForce RTX3060 12GB GPU
* 16 GB RAM
