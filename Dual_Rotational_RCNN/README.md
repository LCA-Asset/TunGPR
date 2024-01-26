The Dual Rotational CNN was adapted from the Region Rotational CNN (R2CNN) proposed by Yang et al (2018), with two major modifications. First, the R2CNN was modified to detect the dual parts of hyperbolas in GPR B-scans. Secondly, the Smooth L1 loss was replaced by the IoU loss. 

To reproduce the Dual Rotational CNN [https://github.com/yangxue0827/R2CNN_FPN_Tensorflow], there are two available options.
1. Refer to the original R2DCNN and replace certain files with the follwing ones in this repository:<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\data\io\read_tfrecord.py<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\libs\configs\cfgs.py<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\libs\fast_rcnn\build_fast_rcnn1.py<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\libs\label_name_dict\label_dict.py<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\libs\losses\losses.py<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\libs\rpn\build_rpn.py<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\tools\train1.py<br>
&nbsp;- TunGPR\Dual_Rotational_RCNN\tools\eval1.py<br>
2. Also, you can git clone this repository and implement the project in your machine. <br>
<br>
While both options work, the first one is recommended as configuring environment and setting up may be a hustle, and it is recommended to start from the scratch. 