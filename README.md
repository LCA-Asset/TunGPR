# TunGPR
A framework, TunGPR, is proposed for Ground Penetrating Radar (GPR)-based tunnel lining assessment by incorporating synthetic database, deep learning-enabled interpretation, and Building Information Modelling (BIM)-based management. In this repository, the TunGPR synthetic dataset, the Dual Rotational CNN, and the Control Point-guided (CPS) algorithm are shared. 
## Contents
* Overview of the TunGPR framework
* Architecture of the TunGPR dataset
* End-to-end Hyperbola fitting
* Representative Results
* Citation
## Overview of the TunGPR framework 
This framework integrates ground information with an as-built BIM-based tunnel model, which allows for direct use of existing geometric and semantic information. Via a dedicated digital workflow, the conceived models in HDF5 format are used to construct dielectric models for FDTD simulation by using an open-source software gprMax (Warren et al. 2016). The GPR modelling employs the probabilistic domain randomisation to generate dielectric models across a wide range of scenarios, then processed through batch FDTD simulations to create the TunGPR dataset. A modified dual-rotational CNN (Jiang et al. 2017), is employed to facilitate hyperbola detection. The abundancy of diversity and quantity of the dataset promotes the learning process by integrating the data-driven algorithm into an end-to-end workflow, which achieves detecting, segmenting, and fitting of hyperbolic signatures at the limit of shape irregularities and background interferences. The last component communicates insights based on results of a probabilistic and/or deterministic risk assessment. This module enables BIM-based hotspot analysis with the risk levels manifested via colour coding for holistic data management.
![overview](https://github.com/LCA-Asset/TunGPR/assets/153473488/1fd21315-438a-43ed-ae02-e7f5eda179d6) <br>

## Architecture of the TunGPR dataset
The dataset constitutes synthetic data produced through FDTD simulations. Featuring a wide range of lining characteristics, the synthetic portion of the dataset includes over 2,000 cases showcasing diverse features. The features include surface roughness (R), delamination (D), dry voids (Vd), saturated voids (Vs), dry cavities (Cd), and saturated cavities (Cs), each catalogued to facilitate detection and classification tasks. Initially configured in HDF5 format to represent complex geometries, these dielectric models were then imported into gprMax for simulation. Leveraging GPU acceleration, the simulations were processed in a batch, leading to the creation of B-scans, which were subsequently saved as grey images (resolution of 512×512×1) with the last channel converting the signal amplitude into greyscale value. The built-in resize function in OpenCV was used to reshape the profiles with the bilinear interpolation method employed. The integration of dielectric model ensures the automatic generation of bounding box labels, providing references for features observed in the radargrams. As an example, the original output B-scan was saved as greyscale images (.png) along with corresponding annotations (.xml), serves the development and testing of data-driven algorithms for image-based tunnel linings. This dataset also offers detailed gprMax files including geometry (.h5), material properties (.txt), simulation commands (.in), and original B-scan outputs (.out), enabling users to either replicate the given models or adapt them to suit specific requirements. <br>
![2](https://github.com/LCA-Asset/TunGPR/assets/153473488/3a0e0faa-fe12-458a-bd45-c3d06873e937) <br>
 
### Dataset 1 - Tunnel lining
This dataset concentrates on mapping complete profiles of tunnel linings, while the second segment is dedicated to the analysis of hyperbolic signatures. Part I of synthetic dataset includes a detailed collection of tunnel lining profiles, featuring over 1,000 examples of intact linings (group R) and approximately 550 cases of damaged linings with issues like delamination, saturated voids, or dry voids. These models were simulated using Ricker wave with the central frequency of 800 MHz, with the voxel size of 4 mm, the time window of 30 ns, and the trace distance of 12 mm. The dataset offers a comprehensive view of various tunnel lining conditions, organised into six groups. Due to the domain randomisation of material properties, it should be noted that in certain cases, the interlayer reflections may not be visually distinguishable. This is primarily due to the low contrast in relative permittivity between concrete and grout.  Additionally, the presence of rough interlayers in these models introduces more interference in object detection tasks compared to the regular model, demonstrating the complexities involved in interpreting GPR data from irregular tunnel linings. 
Download link: [https://drive.google.com/file/d/1OojzmbI5tDJrUtShyAYzcVTZc-lHmTLN/view?usp=sharing] <br>
The folder structure is as the above figure. <br> 

![3](https://github.com/LCA-Asset/TunGPR/assets/153473488/a3825bee-3320-4646-87ac-c25835c97308)
 <br>

### Dataset 2 - Hyperbolic feature
This dataset contains 1,000 different tunnel lining defects that generate hyperbolic features. Their simulations input files and output results are included. In total of 537 cases were randomly selected from the dataset to pretrain the Dual Rotational CNN. To enhance the similarity between the simulated data and field recordings, noises were added to the simulated B-scans. Annotations of the dual hyperbolas of each B-scan are also included in the dataset. <br>
Download link: [https://drive.google.com/file/d/10Y3EEOUUtwHZY1K41uZUN3NDbQn4_uXe/view?usp=sharing] <br>
The folder structure is as below. <br>
├── B-scans/<br>
│ ├── `.png` or `.jpg` files x 537<br>
├── Annotation/<br>
│ ├── `.xml`files x 537<br>
└── gprMax files/<br>
│ ├── Geometry/<br>
│   ├── `.h5` files x 1,000<br>
│ └── Material/<br>
│   ├── `.txt` files x 1,000<br>
│ └── Simulation/<br>
│   ├── in/<br>
│     ├── `.in` files x 1,000<br>
│   ├── out/<br>
│     ├──  `merged.out` files x 1,000<br>

![](C:\Users\zhuhu\Desktop\Hyperbolic.png)

## End-to-end Hyperbola fitting 
In the proposed framework, a dual-rotational CNN model, modified from the R2CNN (Jiang et al. 2017), is utilised to detect hyperbolas, as depicted in Figure 10. The dual-rotational CNN retains the fundamental structure of Faster R-CNN. The detection process starts with the Region Proposal Network (RPN) generating 256 axis-aligned bounding boxes. These proposals, along with their corresponding feature maps, are then passed to the ROI pooling layer. Subsequently, a two-layer Fully Connection (FC) network predicts regression parameters to adjust these boxes to inclined ones. Finally, the Non-Maximum Suppression (NMS) algorithm is applied to select the best proposal for each class (left or right hyperbola in this case). For classification tasks in the dual-rotational CNN, the SoftMax cross entropy loss was employed. As for the regression tasks, the smooth L1 loss, which was originally used in R2CNN, was initially considered. However, due to the limitation of scale invariance in the smooth L1 loss, this study opted for the IoU loss (Yu et al. 2016a) instead.
![workflow](https://github.com/LCA-Asset/TunGPR/assets/153473488/baa5181a-8fcf-4abb-af21-917e9ef37ca5)

After locating the hyperbolic area, the ROI is extracted from the GPR B-scans, and subsequent post-processing steps are conducted within this refined region. Firstly, the OTSU thresholding is applied to binarise the ROI, followed by the CPS algorithm to remove outliers. <br>

![cps](https://github.com/LCA-Asset/TunGPR/assets/153473488/c78125b3-905e-4289-b9b6-848ba324836e)


## Representative Results
### Dual Rotaional CNN VS Faster RCNN
![4](https://github.com/LCA-Asset/TunGPR/assets/153473488/2418a01f-baff-4de7-b3cb-2e75662d80ed)

## Citation
Please cite: 
Huamei Zhu, Mengqi Huang, Qian-Bing Zhang. *TunGPR: Enhancing data-driven maintenance for tunnel linings through synthetic datasets, deep learning and BIM.* Tunnelling and Underground Space Technology. 2024, 145, 105568. [https://doi.org/10.1016/j.tust.2023.105568.]
