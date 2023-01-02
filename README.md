# Deep transfer learning for mass segmentation and classification from screen film mammograms

This repository includes breast cancer mass segmentation investigation using U-net++Xception and other U-net based segmentation models. The first and second steps involve breast segmentation and mass segmentation respectively. The two step enabled to focus only on the mass segmentation in the second step, which is a critical task in breast cancer diagnosis. Finally, in the third step, the mass segmentation predictions were classified as benign versus malignant.

The five-layer U-net was found to have the highest breast segmentation performance. The mammogram pixel dimensions were 1024x768. The segmentation performance was sufficient and other deep transfer learning methods were not investigated. This step enabled segmentation of breast tissue and discarding the mammogram background. 

## Breast segmentation
Breast segmentation training and validation progress can be seen from the following notebook:

https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_unet5L_mg_seg04_4nov22_v001.ipynb

To test the breast segmentation model, download the breast segmentation five-layer U-net model from the following link:
https://drive.google.com/file/d/1r-Qbv4nqxLVIOX09bt7YQKEqN0ArpF_3/view?usp=sharing

Run the following codes for 1024x768 mammograms by locating the model and mammograms in appropriate folders: (Here example is given for CBIS-DDSM mammograms)

https://github.com/tiryakiv/Mass-segmentation/blob/main/test_breast_seg_cbis_mass_cc_mg_test_2dec22.ipynb

## Mass segmentation
After the first step, the mammogram background noise sources were removed, the blank regions were removed, and mammograms were downsized to 640x640. The downsampling enabled training of large deep transfer learning segmentation models.

In the second step,  the mass segmentation performances of the newly proposed U-net++Xception and other recent U-net based nine models were investigated. The newly proposed U-net++Xception model has better performance than U-net5L, Unet++, ResUnet, DeepLabV3Plus and AttentionU-net in terms of DSC. 

Mass segmentation training and validation progress can be seen from the following notebooks:

https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_ResUnet_mass_seg08_14nov22_v011.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_basic_AUnet_mass_seg08_18nov22_v017.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_deepLabV3plus_mass_seg08_17nov22_v015.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_multiResUnet_mass_seg08_17nov22_v016.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_uNetPlusPlusXcept_mass_seg08_16nov22_v013.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_uResNet50_mass_seg08_12nov22_v009.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_uVGG16_mass_seg08_10nov22_v007.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_uXception_mass_seg08_9nov22_v006.ipynb
https://github.com/tiryakiv/Mass-segmentation/blob/main/train_valid_unet5L_mass_seg08_7nov22_v001.ipynb


Model performance progress files for each model at each epoch are shared under model_performance folder. Each file starts with 'data' (as given in the above train_valid... progress codes) and continue with the model number and followed by cross-validation fold number. For example, 'data_v013_2' is the performance file for model 13, which is U-net++Xception, and cross-validation fold number is 2. You can see the model number and model type correspondence from the 'train_valid' codes given above. The purpose of the cross-validation is to see the performance of each model in the entire training dataset.

Mass segmentation validation and test performance results of U-net++Xception (trained with BCDR as explained in the manuscript) in terms of DSC and AUC on BCDR mammograms can be seen from these notebooks: 

https://github.com/tiryakiv/Mass-segmentation/blob/main/valid_mass_seg_predictions_19dec22.ipynb

https://github.com/tiryakiv/Mass-segmentation/blob/main/test_mass_seg_predictions_16dec22.ipynb

Mass segmentation test performance of U-net++Xception (trained with BCDR as explained in the manuscript) in terms of DSC and AUC on 170 CBIS-DDSM test mammograms:

https://github.com/tiryakiv/Mass-segmentation/blob/main/test_cbis_mass_cc_seg_predictions_18dec22.ipynb

To test the performance of Unet++Xception model for mass segmentation on your own mammograms, please download the following mass segmentation test code:
https://github.com/tiryakiv/Mass-segmentation/blob/main/test_uNetPlusPlusXcept_mass_seg08_23nov22_v013.ipynb

You can download the Unet++Xception model of cross-validation number 5 from the following link:

https://drive.google.com/file/d/1UGI08AFreky2ArKJJpa93UZB_iFtOmKR/view?usp=sharing

Then move the model to the "files_mass_seg_xval" folder.  Locate your input mammograms under "mass_seg_08/test/pred" folder, and then execute the above code. Note that the input mammograms should have 640x640 resolution.

## Mass classification
In the third and final step the mass segmentation model predictions were classified into benign versus malignant. The purpose of this step was to demonstrate the entire system performance for automated breast cancer diagnosis. Mass classification train-valid-test codes, performance results on validation, and test results for VGG16:

https://github.com/tiryakiv/Mass-segmentation/blob/main/mass_class_inceptionv3_xval_28nov22.ipynb

https://github.com/tiryakiv/Mass-segmentation/blob/main/mass_class_resnet50v2_xval_28nov22.ipynb

https://github.com/tiryakiv/Mass-segmentation/blob/main/mass_class_vgg16_xval_28nov22.ipynb

https://github.com/tiryakiv/Mass-segmentation/blob/main/mass_class_vgg16_valid_28nov22.ipynb

https://github.com/tiryakiv/Mass-segmentation/blob/main/mass_class_vgg16_test_28nov22.ipynb

Data code names start with 'data' and followed by channel number. Breast segmentation data file is set at 1024x768 resolution while mass segmentation data file is set at  640x640 resolution. 
Deep learning model code names start with "model" and followed by the model name. The references for models are given in the top line as a comment and References section of this page. Deep learning segmentation model code names include the date of experiment and followed by version number. Version number is required to distinguish between models. Batch size is two due to the memory limit and to establish fairness for comparing models' performance.

To run the codes yourself, download all of them. Download the mammograms from the BCDR website: https://bcdr.eu/, apply breast segmentation, and downsample the mammograms and distribute the mammograms to train, valid, and test folders. You can download the CBIS-DDSM mammograms from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629. The BCDR train, validation, testing patient ids are provided in the Supplementary Materials document. Folder hiearchy for mass segmentation is shown below. 

## Training, validation, and testing folder hierarchy for mass segmentation five-fold cross-validation: 
(mg, mask, pred folders denote mammogram, ground truth mask, and prediction folders respectively. For each cross-validation fold, I needed to re-start the kernel to initialize the He normal algorithm appropriately.)
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
* CUDA version 11.8; Driver Version: 522.06, cudatoolkit 11.2.2; cudnn 8.1.0.77
* ipykernel

## Hardware configuration:
* Dell WS T7610
* Intel Xeon E5-2630 2.6 GHz CPU
* GeForce RTX3060 12GB GPU
* 16 GB RAM


## References:
[1]	L. Shen, L.R. Margoiles, J.H. Rothstein, E. Fluder, R. McBride, W. Sieh, Deep Learning to improve Breast cancer Detection on Screening Mammography, Sci. Rep. 9 (2019) 1–12. https://doi.org/https://doi.org/10.1038/s41598-019-48995-4.

[2]	O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical image segmentation, Lect. Notes Comput. Sci. (Including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics). 9351 (2015) 234–241. https://doi.org/10.1007/978-3-319-24574-4_28.

[3]	Z. Zhou, M.M.R. Siddiquee, N. Tajbakhsh, J. Liang, UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation, IEEE Trans. Med. Imaging. 39 (2020) 1856–1867. https://doi.org/10.1109/TMI.2019.2959609.

[4]	L.C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam, Encoder-decoder with atrous separable convolution for semantic image segmentation, Lect. Notes Comput. Sci. (Including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics). 11211 LNCS (2018) 833–851. https://doi.org/10.1007/978-3-030-01234-2_49.

[5]	A. Baccouche, B. Garcia-Zapirain, C. Castillo Olea, A.S. Elmaghraby, Connected-UNets: a deep learning architecture for breast mass segmentation, Npj Breast Cancer. 7 (2021) 1–12. https://doi.org/10.1038/s41523-021-00358-x.

[6]	N.K. Tomar, A. Shergill, B. Rieders, U. Bagci, D. Jha, TransResU-Net: Transformer based ResU-Net for Real-Time Colonoscopy Polyp Segmentation, ArXiv. (2022) 1–4. http://arxiv.org/abs/2206.08985.

[7]	F. Chollet, Xception: Deep learning with depthwise separable convolutions, Proc. - 30th IEEE Conf. Comput. Vis. Pattern Recognition, CVPR 2017. 2017-Janua (2017) 1800–1807. https://doi.org/10.1109/CVPR.2017.195.

[8]	M.A.G. López, N.G. de Posada, D.C. Moura, R.R. Pollán, J.M.F. Valiente, C.S. Ortega, M.R. del Solar, G.D. Herrero, I.M.A.P. Ramos, J.P. Loureiro, T.C. Fernandes, B.M.F. de Araújo, BCDR : A BREAST CANCER DIGITAL REPOSITORY, in: 15th Int. Conf. Exp. Mech., Porto/Portugal, 2012: pp. 1–5.

[9]	L.R. Dice, Measures of the Amount of Ecologic Association Between Species, Ecology. 26 (1945) 297–302. https://doi.org/10.2307/1932409.

[10]	I.C. Moreira, I. Amaral, I. Domingues, A. Cardoso, M.J. Cardoso, J.S. Cardoso, INbreast: Toward a Full-field Digital Mammographic Database, Acad. Radiol. 19 (2012) 236–248. https://doi.org/10.1016/j.acra.2011.09.014.

[11]	BCDR - Breast Cancer Digital Repository, (2012). http://bcdr.inegi.up.pt.

[12]	D.C. Moura, M.A.G. López, P. Cunha, N.G. de Posada, R.R. Pollan, I. Ramos, J.P. Loureiro, I.C. Moreira, B.M.F. de Araújo, T.C. Fernandes, Benchmarking Datasets for Breast Cancer Computer-Aided Diagnosis (CADx), in: J. Ruiz-Shulcloper, G. di Baja (Eds.), Prog. Pattern Recognition, Image Anal. Comput. Vision, Appl., Springer Berlin Heidelberg, Berlin, Heidelberg, 2013: pp. 326–333.

[13]	R. Ramos-Pollán, M.A. Guevara-López, C. Suárez-Ortega, G. Díaz-Herrero, J.M. Franco-Valiente, M. Rubio-del-Solar, N. González-de-Posada, M.A.P. Vaz, J. Loureiro, I. Ramos, Discovering Mammography-based Machine Learning Classifiers for Breast Cancer Diagnosis, J. Med. Syst. 36 (2012) 2259–2269. https://doi.org/10.1007/s10916-011-9693-2.

[14]	M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G.S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mane, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viegas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, X. Zheng, TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems, in: Proc. 12th USENIX Conf. Oper. Syst. Des. Implement., USENIX Association, Savannah, GA, USA, 2016: pp. 265–283. http://arxiv.org/abs/1603.04467.

[15]	F. and others Chollet, Keras, GitHub. (2015). https://keras.io (accessed October 30, 2021).

[16]	D.P. Kingma, J.L. Ba, Adam: A method for stochastic optimization, 3rd Int. Conf. Learn. Represent. ICLR 2015 - Conf. Track Proc. (2015) 1–15.

[17]	K. He, X. Zhang, S. Ren, J. Sun, Delving deep into rectifiers: Surpassing human-level performance on imagenet classification, in: 2015 IEEE Int. Conf. Comput. Vis., 2015: pp. 1026–1034. https://doi.org/10.1109/ICCV.2015.123.

[18]	S. Ioffe, C. Szegedy, Batch normalization: Accelerating deep network training by reducing internal covariate shift, 32nd Int. Conf. Mach. Learn. ICML. 1 (2015) 448–456.

[19]	A.L. Maas, A.Y. Hannun, A.Y. Ng, Rectifier nonlinearities improve neural network acoustic models, Proc. 30th Int. Conf. Mach. Learn. 30 (2013).

[20]	N. Srivastava, G.E. Hinton, A. Krizhevsky, I. Salakhutdinov, R. Salakhutdinov, Dropout: A Simple Way to Prevent Neural Networks from Overfittin, J. Mach. Learn. Res. 15 (2014) 1929–1958. http://jmlr.org/papers/v15/srivastava14a.html.

[21]	HZCTony, Unet : multiple classification using Keras, Github. (2019). https://github.com/HZCTony/U-net-with-multiple-classification.

[22]	C.H. Sudre, W. Li, T. Vercauteren, S. Ourselin, M. Jorge Cardoso, Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations, Lect. Notes Comput. Sci. (2017) 240–248. https://doi.org/10.1007/978-3-319-67558-9_28.

[23]	Z. Zhang, Q. Liu, Y. Wang, Road Extraction by Deep Residual U-Net, IEEE Geosci. Remote Sens. Lett. 15 (2018) 749–753. https://doi.org/10.1109/LGRS.2018.2802944.

[24]	H. Sun, C. Li, B. Liu, Z. Liu, M. Wang, H. Zheng, D.D. Feng, S. Wang, {AUNet}: attention-guided dense-upsampling networks for breast mass segmentation in whole mammograms, Phys. Med. {\&} Biol. 65 (2020) 55005. https://doi.org/10.1088/1361-6560/ab5745.

[25]	O. Oktay, J. Schlemper, L. Le Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N.Y. Hammerla, B. Kainz, B. Glocker, D. Rueckert, Attention U-Net: Learning Where to Look for the Pancreas, in: 1st Conf. Med. Imaging with Deep Learn. (MIDL 2018), Amsterdam, 2018. http://arxiv.org/abs/1804.03999.

[26]	N. Ibtehaz, M.S. Rahman, MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation, Neural Networks. 121 (2020) 74–87. https://doi.org/10.1016/j.neunet.2019.08.025.

[27]	N. Tomar, Semantic-Segmentation-Architecture/TensorFlow/, Github. (2022). https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/tree/main/TensorFlow (accessed February 1, 2022).

[28]	S. Jadon, A survey of loss functions for semantic segmentation, in: 2020 IEEE Conf. Comput. Intell. Bioinforma. Comput. Biol. CIBCB 2020, Via del Mar, Chile, 2020: pp. 1–7. https://doi.org/10.1109/CIBCB48159.2020.9277638.

[29]	Jia Deng, Wei Dong, R. Socher, Li-Jia Li, Kai Li, Li Fei-Fei, ImageNet: A large-scale hierarchical image database, in: 2009 IEEE Conf. Comput. Vis. Pattern Recognit., IEEE, Miami, FL, USA, 2009: pp. 248–255. https://doi.org/10.1109/cvprw.2009.5206848.

[30]	K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale image recognition, in: 3rd Int. Conf. Learn. Represent. ICLR 2015 - Conf. Track Proc., 2015: pp. 1–14.

[31]	K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016: pp. 770–778.

[32]	Siddhartha, Unet Xception Keras for Pneumothorax Segmentation, Kaggle. (2019). https://www.kaggle.com/meaninglesslives/unet-xception-keras-for-pneumothorax-segmentation.

[33]	P. Jaccard, The distribution of the flora in the Alpine zone, New Phytol. XI (1912) 37–50. https://doi.org/10.1111/j.1469-8137.1912.tb05611.x.

[34]	B.W. Matthews, Comparison of the predicted and observed secondary structure of T4 phage lysozyme, Biochim. Biophys. Acta - Protein Struct. 405 (1975) 442–451. https://doi.org/10.1016/0005-2795(75)90109-9.

[35]	J. Cohen, A coefficient of agreement for nominal scales, Educ. Psychol. Meas. 20 (1960) 37–46. https://doi.org/10.1177/001316446002000104.

[36]	M.J. Warrens, On the Equivalence of Cohen’s Kappa and the Hubert-Arabie Adjusted Rand Index, J. Classif. 25 (2008) 177–183. https://doi.org/10.1007/S00357-008-9023-7.

[37]	K. He, X. Zhang, S. Ren, J. Sun, Identity mappings in deep residual networks, Lect. Notes Comput. Sci. (Including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics). 9908 LNCS (2016) 630–645. https://doi.org/10.1007/978-3-319-46493-0_38.

[38]	C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna, Rethinking the Inception Architecture for Computer Vision, Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 2016-Decem (2016) 2818–2826. https://doi.org/10.1109/CVPR.2016.308.

[39]	F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, E. Duchesnay, Scikit-learn: Machine Learning in Python, J. Mach. Learn. Res. 12 (2011) 2825–2830.
