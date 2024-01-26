# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os



# root path
ROOT_PATH = os.path.abspath('//Dual_Rotational_RCNN//')

# pretrain weights path
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_result'
TEST_SAVE_PATH1 = ROOT_PATH + '/result'

#NET_NAME = 'vgg16'
NET_NAME = 'resnet_v1_50'
#NET_NAME1 = 'vgg_16'
NET_NAME1 = 'resnet_v1_50'
# VERSION = 'aligned_20221021_real'
VERSION = 'debug'
VERSION1 = 'aligned1'
# VERSION3 = 'try'
# VERSION2 = 'try'

IMG_NUM = 94
IMG_NUM1 = 94
CLASS_NUM = 7
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
STRIDE = [4, 8, 16, 32, 64]  # vgg16
ANCHOR_SCALES = [1., 2., 4., 8.]
ANCHOR_RATIOS = [1 / 3., 1., 3.0]
SCALE_FACTORS = [10., 10., 5., 5., 5.]
OUTPUT_STRIDE = 16
SHORT_SIDE_LEN = 512
DATASET_NAME = 'r_aligned_hyperbola'
IS_VALID = False
BATCH_SIZE = 1
WEIGHT_DECAY = {'resnet_v1_50': 0.0001, 'resnet_v1_101': 0.0001, 'vgg16': 0.0005}
EPSILON = 1e-5
MOMENTUM = 0.9
MAX_ITERATION = 3000
GPU_GROUP = "0"
#LR = 0.0001
LR = 1e-3
IS_GIOU=False

# rpn
SHARE_HEAD = False
RPN_NMS_IOU_THRESHOLD = 0.7
MAX_PROPOSAL_NUM = 500 #300
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
RPN_MINIBATCH_SIZE = 512 #256
RPN_POSITIVE_RATE = 0.5
IS_FILTER_OUTSIDE_BOXES = True
RPN_TOP_K_NMS = 10000
FEATURE_PYRAMID_MODE = 0  # {0: 'feature_pyramid', 1: 'dense_feature_pyramid'}
LOSS = 'iou'

# fast rcnn
ROTATE_NMS_USE_GPU = True
#FAST_RCNN_MODE = 'build_fast_rcnn1'
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 0.5
FAST_RCNN_NMS_IOU_THRESHOLD = 0.3 #0.3
#FAST_RCNN_NMS_IOU_RO_THRESHOLD = 0.5
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 20
FINAL_SCORE_THRESHOLD = 0.8 #0.8
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.3 #0.5
FAST_RCNN_MINIBATCH_SIZE = 128
FAST_RCNN_POSITIVE_RATE = 0.25
