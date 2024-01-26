# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from libs.configs import cfgs

def giou_losses(predict_boxes, gtboxes, g=False):
    epsilon = 0.00001
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gtboxes, axis=1)
    p_ymin, p_xmin, p_ymax, p_xmax = tf.unstack(predict_boxes, axis=1)
    
    gt_area = tf.maximum(epsilon, (gt_xmax-gt_xmin)*(gt_ymax-gt_ymin))
    predict_area = tf.maximum(epsilon, (p_xmax-p_xmin)*(p_ymax-p_ymin))
    
    min_ymax = tf.minimum(gt_ymax, p_ymax)
    max_ymin = tf.maximum(gt_ymin, p_ymin)
    intersect_h = tf.maximum(epsilon, min_ymax-max_ymin)
    min_xmax = tf.minimum(gt_xmax, p_xmax)
    max_xmin = tf.maximum(gt_xmin, p_xmin)
    intersect_w = tf.maximum(epsilon, min_xmax-max_xmin)
    intersect_area = tf.reshape(intersect_w*intersect_h, [-1])
    
    max_ymax = tf.maximum(gt_ymax, p_ymax)
    max_xmax = tf.maximum(gt_xmax, p_xmax)
    min_ymin = tf.minimum(gt_ymin, p_ymin)
    min_xmin = tf.minimum(gt_xmin, p_xmin)
    containing_h = tf.maximum(epsilon, max_ymax-min_ymin)
    containing_w = tf.maximum(epsilon, max_xmax-min_xmin)
    containing_area = tf.reshape(containing_h*containing_w, [-1])
    
    union_area = gt_area+predict_area-intersect_area
    iou = tf.where(tf.equal(intersect_area, 0.0), tf.zeros_like(intersect_area), tf.truediv(intersect_area, union_area))
    unoccupied_area = tf.truediv((containing_area-union_area), tf.maximum(containing_area, epsilon))
#    iou_loss = -tf.math.log(iou)
    giou_loss = 1.0-tf.subtract(iou, unoccupied_area)
    iou_loss = 1.0-iou
    if g:
      return tf.reduce_mean(giou_loss, axis=0)
    else:
      return tf.reduce_mean(iou_loss, axis=0)

def l1_smooth_losses(predict_boxes, gtboxes, object_weights, classes_weights=None):
    '''

    :param predict_boxes: [minibatch_size, -1]
    :param gtboxes: [minibatch_size, -1]
    :param object_weights: [minibatch_size, ]. 1.0 represent object, 0.0 represent others(ignored or background)
    :return:
    '''

    diff = predict_boxes - gtboxes
    abs_diff = tf.cast(tf.abs(diff), tf.float32)

    if classes_weights is None:
        '''
        first_stage:
        predict_boxes :[minibatch_size, 4]
        gtboxes: [minibatchs_size, 4]
        '''
        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(tf.less(abs_diff, 1), 0.5 * tf.square(abs_diff), abs_diff - 0.5),
            axis=1) * object_weights
    else:
        '''
        fast_rcnn:
        predict_boxes: [minibatch_size, 4*num_classes]
        gtboxes: [minibatch_size, 4*num_classes]
        classes_weights : [minibatch_size, 4*num_classes]
        '''
        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(tf.less(abs_diff, 1), 0.5*tf.square(abs_diff)*classes_weights,
                     (abs_diff - 0.5)*classes_weights),
            axis=1)*object_weights
    return tf.reduce_mean(anchorwise_smooth_l1norm, axis=0)  # reduce mean


def weighted_softmax_cross_entropy_loss(predictions, labels, weights):
    '''

    :param predictions:
    :param labels:
    :param weights: [N, ] 1 -> should be sampled , 0-> not should be sampled
    :return:
    # '''
    per_row_cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=predictions,
                                                                labels=labels)

    weighted_cross_ent = tf.reduce_sum(per_row_cross_ent * weights)
    return weighted_cross_ent / tf.reduce_sum(weights)


def test_smoothl1():

    predict_boxes = tf.constant([[1, 1, 2, 2],
                                [2, 2, 2, 2],
                                [3, 3, 3, 3]])
    gtboxes = tf.constant([[1, 1, 1, 1],
                          [2, 1, 1, 1],
                          [3, 3, 2, 1]])

    loss = l1_smooth_losses(predict_boxes, gtboxes, [1, 1, 1])

    with tf.Session() as sess:
        print(sess.run(loss))

if __name__ == '__main__':
    test_smoothl1()
