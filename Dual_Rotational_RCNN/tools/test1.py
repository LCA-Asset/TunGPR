# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from data.io.read_tfrecord import next_batch
from libs.configs import cfgs
from libs.networks.network_factory import get_network_byname
from help_utils.tools import *
from libs.rpn import build_rpn
import cv2

from help_utils import help_utils
from tools import restore_model
from libs.box_utils.coordinate_convert import back_forward_convert
from libs.box_utils.boxes_utils import get_horizen_minAreaRectangle
from libs.fast_rcnn import build_fast_rcnn1
from IPython.core import debugger
debug = debugger.Pdb().set_trace
from libs.box_utils.show_box_in_tensor import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from libs.networks.network_factory import get_flags_byname
FLAGS = get_flags_byname(cfgs.NET_NAME)

def test(img_num):
    with tf.Graph().as_default():

        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.SHORT_SIDE_LEN,
                       is_training=False,
                       is_valid = False)
        gtboxes_and_label = tf.py_func(back_forward_convert,
                                       inp=[tf.squeeze(gtboxes_and_label_batch, 0)],
                                       Tout=tf.float32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 6])

        gtboxes_and_label_minAreaRectangle = get_horizen_minAreaRectangle(gtboxes_and_label)

        gtboxes_and_label_minAreaRectangle = tf.reshape(gtboxes_and_label_minAreaRectangle, [-1, 5])
        with tf.name_scope('draw_gtboxes'):
            gtboxes_in_img = draw_box_with_color(img_batch, tf.reshape(gtboxes_and_label_minAreaRectangle, [-1, 5])[:, :-1],
                                                 text=tf.shape(gtboxes_and_label_minAreaRectangle)[0])

            gtboxes_rotate_in_img = draw_box_with_color_rotate(img_batch, tf.reshape(gtboxes_and_label, [-1, 6])[:, :-1],
                                                               text=tf.shape(gtboxes_and_label)[0])

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=False,
                                          #output_stride=None,
                                          #global_pool=False,
                                          dropout_keep_prob=0.5,
                                          spatial_squeeze=False)

        # ***********************************************************************************************
        # *                                            RPN                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=None,
                            is_training=False,
                            share_head=cfgs.SHARE_HEAD,
                            share_net=share_net,
                            stride=cfgs.STRIDE,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

        # rpn predict proposals
        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]
        with tf.name_scope('draw_proposals'):
            # score > 0.5 is object
            rpn_object_boxes_indices = tf.reshape(tf.where(tf.greater(rpn_proposals_scores, 0.5)), [-1])
            rpn_object_boxes = tf.gather(rpn_proposals_boxes, rpn_object_boxes_indices)

            rpn_proposals_objcet_boxes_in_img = draw_box_with_color(img_batch, rpn_object_boxes,
                                                                    text=tf.shape(rpn_object_boxes)[0])
            rpn_proposals_boxes_in_img = draw_box_with_color(img_batch, rpn_proposals_boxes,
                                                             text=tf.shape(rpn_proposals_boxes)[0])
        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn1.FastRCNN(img_batch,feature_pyramid=rpn.feature_pyramid,
                                              rpn_proposals_boxes=rpn_proposals_boxes,
                                              rpn_proposals_scores=rpn_proposals_scores,
                                              img_shape=tf.shape(img_batch),
                                              roi_size=cfgs.ROI_SIZE,
                                              roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                              scale_factors=cfgs.SCALE_FACTORS,
                                              gtboxes_and_label=None,
                                              gtboxes_and_label_minAreaRectangle=gtboxes_and_label_minAreaRectangle,
                                              fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                              fast_rcnn_maximum_boxes_per_img=100,
                                              fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                              show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                              # show detections which score >= 0.6
                                              num_classes=cfgs.CLASS_NUM,
                                              fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                              fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                              fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                              # iou>0.5 is positive, iou<0.5 is negative
                                              use_dropout=cfgs.USE_DROPOUT,
                                              weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                              is_training=False,
                                              level=cfgs.LEVEL)

        #fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category, \
        fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, num_of_objects_rotate, detection_category_rotate = \
            fast_rcnn.fast_rcnn_predict()
        with tf.name_scope('draw_boxes_with_categories'):

            fast_rcnn_predict_rotate_boxes_in_imgs = draw_boxes_with_categories_rotate(img_batch=img_batch,
                                                                                       boxes=fast_rcnn_decode_boxes_rotate,
                                                                                       labels=detection_category_rotate,
                                                                                       scores=fast_rcnn_score_rotate)
        # train
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            for i in range(img_num):

                start = time.time()

                _share_net,_img_name_batch, _img_batch, _gtboxes_and_label, \
                _fast_rcnn_decode_boxes_rotate, \
                _fast_rcnn_score_rotate, _detection_category_rotate \
                    = sess.run([share_net,img_name_batch, img_batch, gtboxes_and_label, fast_rcnn_decode_boxes_rotate,
                                fast_rcnn_score_rotate, detection_category_rotate])
                end = time.time()
                _img_batch = np.squeeze(_img_batch, axis=0)
               

                _img_batch_fpn_rotate = help_utils.draw_rotate_box_cv(_img_batch,
                                                                      boxes=_fast_rcnn_decode_boxes_rotate,
                                                                      labels=_detection_category_rotate,
                                                                      scores=_fast_rcnn_score_rotate)
                mkdir(cfgs.TEST_SAVE_PATH)
                #cv2.imwrite(cfgs.TEST_SAVE_PATH + '/{}_horizontal_fpn.jpg'.format(str(_img_name_batch[0])), _img_batch_fpn_horizonal)
                cv2.imwrite(cfgs.TEST_SAVE_PATH + '/{}_rotate_fpn.jpg'.format(str(_img_name_batch[0])), _img_batch_fpn_rotate)
          
                temp_label_horizontal = np.reshape(_gtboxes_and_label[:, -1:], [-1, ]).astype(np.int64)
                temp_label_rotate = np.reshape(_gtboxes_and_label[:, -1:], [-1, ]).astype(np.int64)

                

                _img_batch_gt_rotate = help_utils.draw_rotate_box_cv(_img_batch,
                                                                     boxes=_gtboxes_and_label[:, :-1],
                                                                     labels=temp_label_rotate,
                                                                     scores=None)

                #cv2.imwrite(cfgs.TEST_SAVE_PATH + '/{}_horizontal_gt.jpg'.format(str(_img_name_batch[0])), _img_batch_gt_horizontal)
                #cv2.imwrite(cfgs.TEST_SAVE_PATH + '/{}_rotate_gt.jpg'.format(str(_img_name_batch[0])), _img_batch_gt_rotate)

                view_bar('{} image cost {}s'.format(str(_img_name_batch[0]), (end - start)), i + 1, img_num)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    img_num =21
    test(img_num)










