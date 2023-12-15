# TunGPR: Enhancing Data-Driven Maintenance for Tunnel Linings through Synthetic Datasets, Deep Learning and BIM
A framework is proposed, namely TunGPR, for Ground Penetrating Radar (GPR)-based tunnel lining assessment by incorporating Building Information Modelling (BIM), synthetic database and deep learning-enabled interpretation. In this repository, the TunGPR synthetic dataset of tunnel lining and hyperbolic features are shared, as well as the Dual Rotational CNN, trained to detect the hyperbolas in GPR images with two rotational bounding boxes. <be>`This repository is to be updated with more details. Datasets, codes, and instructions will be uploaded shortly. Please stay tuned.` 
## Contents
* Overview
* Representative Results
* Dataset
* End-to-end Hyperbola Fitting
* Citation
## Overview

## Architecture of the TunGPR dataset
The dataset constitutes synthetic data produced through FDTD simulations. Featuring a wide range of lining characteristics, the synthetic portion of the dataset includes over 2,000 cases showcasing diverse features. The features include surface roughness (R), delamination (D), dry voids (Vd), saturated voids (Vs), dry cavities (Cd), and saturated cavities (Cs), each catalogued to facilitate detection and classification tasks. Initially configured in HDF5 format to represent complex geometries, these dielectric models were then imported into gprMax for simulation. Leveraging GPU acceleration, the simulations were processed in a batch, leading to the creation of B-scans, which were subsequently saved as grey images (resolution of 512×512×1) with the last channel converting the signal amplitude into greyscale value. The built-in resize function in OpenCV was used to reshape the profiles with the bilinear interpolation method employed. The integration of dielectric model ensures the automatic generation of bounding box labels, providing references for features observed in the radargrams. As an example, the original output B-scan was saved as greyscale images (.png) along with corresponding annotations (.xml), serves the development and testing of data-driven algorithms for image-based tunnel linings. This dataset also offers detailed gprMax files including geometry (.h5), material properties (.txt), simulation commands (.in), and original B-scan outputs (.out), enabling users to either replicate the given models or adapt them to suit specific requirements. <br>
![image](https://github.com/LCA-Asset/TunGPR/assets/153473488/60c1f452-0b13-4bb6-acf0-0c9cadaf6c5f)
## Representative Results
![image](https://github.com/LCA-Asset/TunGPR/assets/153473488/20cbbbd8-16d8-4295-ae13-72fc6f8cf017)
## Dataset
The dataset of `Tunnel lining` can be downloaded here. <br>
<br>The dataset of `Hyperbolic feature` can be downloaded here.
## End-to-end Hyperbola Fitting
- Dual rotational CNN for detection
- Post-processing for segmentation
## Citation
