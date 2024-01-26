# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')
from IPython.core import debugger
debug = debugger.Pdb().set_trace
import tensorflow as tf
import os
import time
import cv2
from help_utils.tools import *
from help_utils import help_utils
from data.io.read_tfrecord import next_batch
from libs.networks.network_factory import get_network_byname
from libs.label_name_dict.label_dict import *
from libs.rpn import build_rpn
from help_utils.tools import view_bar
from tools import restore_model
import pickle
from libs.box_utils.coordinate_convert import *
from libs.box_utils.boxes_utils import get_horizen_minAreaRectangle
from libs.fast_rcnn import build_fast_rcnn1
from libs.box_utils import iou_rotate
import matplotlib.pyplot as plt
from collections import OrderedDict
import tensorflow.contrib.slim as slim
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def make_dict_packle(_gtboxes_and_label, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category):

    gtbox_list = []
    predict_list = []

    for j, box in enumerate(_gtboxes_and_label):
        bbox_dict = {}
        bbox_dict['bbox'] = np.array(_gtboxes_and_label[j, :-1], np.float64)
        bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label[j, -1])]
        gtbox_list.append(bbox_dict)
        #debug()
    for label in NAME_LABEL_MAP.keys():
        
        if label == 'back_ground':
            continue
        else:
            temp_dict = {}
            temp_dict['name'] = label

            ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
            temp_boxes = _fast_rcnn_decode_boxes[ind]
            temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
            temp_dict['bbox'] = np.array(np.concatenate([temp_boxes, temp_score], axis=1), np.float64)
            predict_list.append(temp_dict)
    return gtbox_list, predict_list


def eval_ship(img_num, mode):
    with tf.Graph().as_default():

        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.SHORT_SIDE_LEN,
                       is_training=False,
                       is_valid = cfgs.IS_VALID)

        gtboxes_and_label = tf.py_func(back_forward_convert,
                                       inp=[tf.squeeze(gtboxes_and_label_batch, 0)],
                                       Tout=tf.float32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 6])

        gtboxes_and_label_minAreaRectangle = get_horizen_minAreaRectangle(gtboxes_and_label)

        gtboxes_and_label_minAreaRectangle = tf.reshape(gtboxes_and_label_minAreaRectangle, [-1, 5])

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        logits, share_net = get_network_byname(net_name=cfgs.NET_NAME,
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
                            share_net=logits,
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
        rpn_proposals_boxes, rpn_proposals_scores= rpn.rpn_proposals()  # rpn_score shape: [300, ]
        #feature_maps_dict = get_feature_maps(net_name=cfgs.NET_NAME,share_net= share_net )
        #feature_pyramid = build_feature_pyramid(feature_maps_dict)
        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn1.FastRCNN(img_batch,feature_pyramid=logits,
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

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category, \
        fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, num_of_objects_rotate, detection_category_rotate = \
            fast_rcnn.fast_rcnn_predict()
        fast_rcnn_decode_boxes_hrotate = get_horizen_minAreaRectangle(fast_rcnn_decode_boxes_rotate, False)

        if mode == 0:
            fast_rcnn_decode_boxes_rotate = get_horizen_minAreaRectangle(fast_rcnn_decode_boxes_rotate, False)

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

            gtboxes_horizontal_dict = {}
            predict_horizontal_dict = {}
            gtboxes_rotate_dict = {}
            predict_rotate_dict = {}

            for i in range(img_num):

                start = time.time()

                _fast_rcnn_decode_boxes_hrotate,_img_name_batch, _img_batch,_gtboxes_and_label,  _fast_rcnn_decode_boxes_rotate, \
                _fast_rcnn_score_rotate, _detection_category_rotate\
                    = sess.run([fast_rcnn_decode_boxes_hrotate,img_name_batch,img_batch,gtboxes_and_label,  fast_rcnn_decode_boxes_rotate,
                                fast_rcnn_score_rotate, detection_category_rotate])
                #for j in range(2,6,1):
                    
                #    _feature_maps_dict['C' + str(j)] = np.squeeze(_feature_maps_dict['C' + str(j)])
                #    plt.figure()
                #    _feature_maps_dict['C' + str(j)] = _feature_maps_dict['C' + str(j)].sum(axis=2)
                #    plt.imshow(_feature_maps_dict['C' + str(j)])
                #    plt.colorbar()
                #    plt.savefig('/files/stu10_file/code/vgg16fpn/tools/xunlian/_{}img_C{}.png'.format(str(_img_name_batch[0]),j))
                #    _feature_pyramid['P' + str(j)] = np.squeeze(_feature_pyramid['P' + str(j)])
                #    plt.figure()
                #    _feature_pyramid['P' + str(j)] = _feature_pyramid['P' + str(j)].sum(axis=2)
                #    plt.imshow(_feature_pyramid['P' + str(j)])
                #    plt.colorbar()
                #    plt.savefig('/files/stu10_file/code/vgg16fpn/tools/xunlian/_{}img_P{}.png'.format(str(_img_name_batch[0]),j))
                end = time.time()
                _img_batch = np.squeeze(_img_batch, axis=0)
               

                _img_batch_fpn_rotate, box_horizontal, bi, mask, rotate_box_position = help_utils.draw_rotate_box_cv(_img_name_batch,_img_batch,
                                                                      boxes=_fast_rcnn_decode_boxes_rotate,
                                                                      hboxes = _fast_rcnn_decode_boxes_hrotate,
                                                                      labels=_detection_category_rotate,
                                                                      scores=_fast_rcnn_score_rotate)
                mkdir(cfgs.TEST_SAVE_PATH+cfgs.VERSION)
                #cv2.imwrite(cfgs.TEST_SAVE_PATH + '/{}_horizontal_fpn.jpg'.format(str(_img_name_batch[0])), _img_batch_fpn_horizonal)
                cv2.imwrite(cfgs.TEST_SAVE_PATH + cfgs.VERSION + '/{}_rotate_fpn.jpg'.format(str(_img_name_batch[0])), _img_batch_fpn_rotate)
                mkdir(cfgs.TEST_SAVE_PATH+cfgs.VERSION+'/segmentation/')
                cv2.imwrite(cfgs.TEST_SAVE_PATH + cfgs.VERSION +'/segmentation/' + '/{}_rotate_fpn.jpg'.format(str(_img_name_batch[0])), bi)
                mkdir(cfgs.TEST_SAVE_PATH+cfgs.VERSION+'/mask/')
                np.save(cfgs.TEST_SAVE_PATH + cfgs.VERSION +'/mask/' + '/{}_rotate_fpn.npy'.format(str(_img_name_batch[0])), mask)
                mkdir(cfgs.TEST_SAVE_PATH+cfgs.VERSION+'/horizontal/')
                cv2.imwrite(cfgs.TEST_SAVE_PATH + cfgs.VERSION +'/horizontal/' + '/{}.jpg'.format(str(_img_name_batch[0])), box_horizontal)
                mkdir(cfgs.TEST_SAVE_PATH+cfgs.VERSION+'/box_position/')
                np.save(cfgs.TEST_SAVE_PATH + cfgs.VERSION +'/box_position/' + '/{}.npy'.format(str(_img_name_batch[0])), rotate_box_position)
               
                gtboxes_rotate_dict[str(_img_name_batch[0])] = []
                
                predict_rotate_dict[str(_img_name_batch[0])] = []

              

                if mode == 0:
                    gtbox_rotate_list, predict_rotate_list = \
                        make_dict_packle(_gtboxes_and_label_minAreaRectangle, _fast_rcnn_decode_boxes_rotate,
                                         _fast_rcnn_score_rotate, _detection_category_rotate)
                else:
                    gtbox_rotate_list, predict_rotate_list = \
                        make_dict_packle(_gtboxes_and_label, _fast_rcnn_decode_boxes_rotate,
                                         _fast_rcnn_score_rotate, _detection_category_rotate)
                #debug()
                
                gtboxes_rotate_dict[str(_img_name_batch[0])].extend(gtbox_rotate_list)
                predict_rotate_dict[str(_img_name_batch[0])].extend(predict_rotate_list)
                #debug()
                view_bar('{} image cost {}s'.format(str(_img_name_batch[0]), (end - start)), i + 1, img_num)

           
            fw3 = open('gtboxes_rotate_dict.pkl', 'wb+')
            fw4 = open('predict_rotate_dict.pkl', 'wb+')
          
            pickle.dump(gtboxes_rotate_dict, fw3)
            pickle.dump(predict_rotate_dict, fw4)
         
            fw3.close()
            fw4.close()
            coord.request_stop()
            coord.join(threads)


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_single_label_dict(predict_dict, gtboxes_dict, label):
    rboxes = {}
    gboxes = {}
    rbox_images = list(predict_dict.keys())
    #rbox_images = list(gtboxes_dict.keys())
    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        gboxes[rbox_image] = []
        for gt_box in gtboxes_dict[rbox_image]:
            if gt_box['name'] == label:
               gboxes[rbox_image].append(gt_box)
        for pre_box in predict_dict[rbox_image]:
            if pre_box['name'] == label:            
               rboxes[rbox_image] = [pre_box]
    return rboxes, gboxes


def eval(rboxes, gboxes, iou_th, use_07_metric, mode):
    rbox_images = list(rboxes.keys())
    fp = np.zeros(len(rbox_images))
    tp = np.zeros(len(rbox_images))
    box_num = 0
    #debug()
    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        if len(gboxes[rbox_image]) > 0 :
            gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
            box_num = box_num + len(gbox_list)
            gbox_list = np.concatenate((gbox_list, np.zeros((np.shape(gbox_list)[0], 1))), axis=1)
            if len(rboxes[rbox_image][0]['bbox']) > 0:
                rbox_lists = np.array(rboxes[rbox_image][0]['bbox']) 
                confidence = rbox_lists[:, -1]
                box_index = np.argsort(-confidence)
                rbox_lists = rbox_lists[box_index, :]           
                for rbox_list in rbox_lists:
                    if mode == 0:
                        ixmin = np.maximum(gbox_list[:, 0], rbox_list[0])
                        iymin = np.maximum(gbox_list[:, 1], rbox_list[1])
                        ixmax = np.minimum(gbox_list[:, 2], rbox_list[2])
                        iymax = np.minimum(gbox_list[:, 3], rbox_list[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih
                        # union
                        uni = ((rbox_list[2] - rbox_list[0] + 1.) * (rbox_list[3] - rbox_list[1] + 1.) +
                               (gbox_list[:, 2] - gbox_list[:, 0] + 1.) *
                               (gbox_list[:, 3] - gbox_list[:, 1] + 1.) - inters)
                        overlaps = inters / uni
                    else:
                        overlaps = iou_rotate.iou_rotate_calculate1(np.array([rbox_list[:-1]]),
                                                                    gbox_list,
                                                                    use_gpu=False)[0]

                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                    if ovmax > iou_th:
                        if gbox_list[jmax, -1] == 0:
                            tp[i] += 1
                            gbox_list[jmax, -1] = 1
                        else:
                            fp[i] += 1
                    else:
                        fp[i] += 1
            else:
                fp[i] += len(rboxes[rbox_image][0]['bbox'])
            
        elif len(rboxes[rbox_image][0]['bbox']) > 0:
             rbox_lists = np.array(rboxes[rbox_image][0]['bbox'])
             if len(gboxes[rbox_image]) > 0:
                gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
             else:
                  fp[i] += 1
    rec = np.zeros(len(rbox_images))
    prec = np.zeros(len(rbox_images))
    if box_num == 0:
        for i in range(len(fp)):
            if fp[i] != 0:
                prec[i] = 0
            else:
                prec[i] = 1

    else:

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        rec = tp / box_num
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, box_num
if __name__ == '__main__':
    img_num =44
    # 0: horizontal standard 1: rotate standard
    mode = 1
    eval_ship(img_num, mode)

    #fr1 = open('gtboxes_horizontal_dict.pkl', 'rb')
    #fr2 = open('predict_horizontal_dict.pkl', 'rb')
    fr3 = open('gtboxes_rotate_dict.pkl', 'rb')
    fr4 = open('predict_rotate_dict.pkl', 'rb')
    #gtboxes_horizontal_dict = pickle.load(fr1)
    #predict_horizontal_dict = pickle.load(fr2)
    gtboxes_rotate_dict = pickle.load(fr3)
    predict_rotate_dict = pickle.load(fr4)
    
    R, P, AP, F, num = [], [], [], [], []
    R1, P1, AP1, F1, num1 = [], [], [], [], []

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue

        #rboxes, gboxes = get_single_label_dict(predict_horizontal_dict, gtboxes_horizontal_dict, label)
        rboxes1, gboxes1 = get_single_label_dict(predict_rotate_dict, gtboxes_rotate_dict, label)
        #rec, prec, ap, box_num = eval(rboxes, gboxes, 0.5, False, mode=0)
        rboxes1 =  OrderedDict(sorted(rboxes1.items(),key=lambda d:d[0]))
        gboxes1 =  OrderedDict(sorted(gboxes1.items(),key=lambda d:d[0]))
        rec1, prec1, ap1, box_num1 = eval(rboxes1, gboxes1, 0.5, False, mode=mode)
        #debug()
        #recall = rec[-1]
        recall1 = rec1[-1]
        #precision = prec[-1]
        precision1 = prec1[-1]
        #F_measure = (2 * precision * recall) / (recall + precision)
        F_measure1 = (2 * precision1 * recall1) / (recall1 + precision1+1e-10)
       # print('\n{}\tR:{}\tP:{}\tap:{}\tF:{}'.format(label, recall, precision, ap, F_measure))
        print('\n{}\tR:{}\tP:{}\tap:{}\tF:{}'.format(label, recall1, precision1, ap1, F_measure1))
      #  R.append(recall)
       # P.append(precision)
       # AP.append(ap)
       # F.append(F_measure)
       # num.append(box_num)

        R1.append(recall1)
        P1.append(precision1)
        AP1.append(ap1)
        F1.append(F_measure1)
        num1.append(box_num1)


    R1 = np.array(R1)
    P1 = np.array(P1)
    AP1 = np.array(AP1)
    F1 = np.array(F1)
    num1 = np.array(num1)
    weights1 = num1 / np.sum(num1)
    #debug()
    Recall1 = np.sum(R1 * weights1)
    Precision1 = np.sum(P1 * weights1)
    mAP1 = np.sum(AP1 * weights1)
    F_measure1 = np.sum(F1 * weights1)

    #print('\n{}\tR:{}\tP:{}\tmAP:{}\tF:{}'.format('horizontal standard', Recall, Precision, mAP, F_measure))
    print('\n{}\tR:{}\tP:{}\tmAP:{}\tF:{}'.format('rotate standard', Recall1, Precision1, mAP1, F_measure1))


    fr3.close()
    fr4.close()









