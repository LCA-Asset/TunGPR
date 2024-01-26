# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
import math


def decode_boxes(encode_boxes, reference_boxes, scale_factors=None, name='decode'): #这个函数是把reference_boxes按照encode_boxes的参数调整，第一阶段在rpn中，renference是anchors，第二阶段进入fastrcnn，reference就成了rpn给出的proposal，所以整个网络中，要调整两次预测框
    '''

    :param encode_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param scale_factors: use for scale
    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 4]
    '''

    with tf.variable_scope(name):
        t_ycenter, t_xcenter, t_h, t_w = tf.unstack(encode_boxes, axis=1) #python中0是列，1是行 #encode_boxes是4个调整参数
        if scale_factors: #scale_factors给了四个数，分别是xc，yc，w，h的调整
            t_xcenter /= scale_factors[0]
            t_ycenter /= scale_factors[1]
            t_w /= scale_factors[2]
            t_h /= scale_factors[3]

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1) #anchor的标记方式从这里给出了，四个值对应着ymin, xmin, ymax, xmax

        reference_xcenter = (reference_xmin + reference_xmax) / 2.
        reference_ycenter = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin #这四行是在转换ahchor的标记方式为标准标记方式：中心点坐标和宽、高

        predict_xcenter = t_xcenter * reference_w + reference_xcenter 
        predict_ycenter = t_ycenter * reference_h + reference_ycenter
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h #这四行是把anchor按照encode_boxes平移缩放成目标框了

        predict_xmin = predict_xcenter - predict_w / 2.
        predict_xmax = predict_xcenter + predict_w / 2.
        predict_ymin = predict_ycenter - predict_h / 2.
        predict_ymax = predict_ycenter + predict_h / 2. #这四行是把上步调整后的目标框，转换成坐标标记方式

        return tf.transpose(tf.stack([predict_ymin, predict_xmin,
                                      predict_ymax, predict_xmax]))


def decode_boxes_rotate(encode_boxes, reference_boxes, scale_factors=None, name='decode'): #这一步是按照encode_boxes来平移、缩放、旋转reference_boxes(proposals)，得到rotate_boxes
    '''

    :param encode_boxes:[N, 5]
    :param reference_boxes: [N, 5] .
    :param scale_factors: use for scale
    in the rpn stage, reference_boxes are anchors
    in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 5]
    '''

    with tf.variable_scope(name):
        t_ycenter, t_xcenter, t_h, t_w, t_theta = tf.unstack(encode_boxes, axis=1)
        if scale_factors:
            t_xcenter /= scale_factors[0]
            t_ycenter /= scale_factors[1]
            t_w /= scale_factors[2]
            t_h /= scale_factors[3]
            t_theta /= scale_factors[4]

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)
        reference_x_center = (reference_xmin + reference_xmax) / 2.
        reference_y_center = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin
        reference_theta = tf.ones(tf.shape(reference_xmin)) * -90

        predict_x_center = t_xcenter * reference_w + reference_x_center
        predict_y_center = t_ycenter * reference_h + reference_y_center
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h

        predict_theta = t_theta * 180 / math.pi + reference_theta



        decode_boxes = tf.transpose(tf.stack([predict_y_center, predict_x_center,
                                              predict_h, predict_w, predict_theta]))

        return decode_boxes


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None, name='encode'): #给anchors（unencode_boxes）编码，对标目标boxes(reference_boxes)，计算出每个anchor的调整参数
    '''

    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 4]
    :param reference_boxes: [H*W*num_anchors_per_location, 4]
    :return: encode_boxes [-1, 4]
    '''

    with tf.variable_scope(name):
        ymin, xmin, ymax, xmax = tf.unstack(unencode_boxes, axis=1)

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

        x_center = (xmin + xmax) / 2.
        y_center = (ymin + ymax) / 2.
        w = xmax - xmin
        h = ymax - ymin

        reference_xcenter = (reference_xmin + reference_xmax) / 2.
        reference_ycenter = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin

        reference_w += 1e-8
        reference_h += 1e-8
        w += 1e-8
        h += 1e-8  # to avoid NaN in division and log below

        t_xcenter = (x_center - reference_xcenter) / reference_w
        t_ycenter = (y_center - reference_ycenter) / reference_h
        t_w = tf.log(w / reference_w)
        t_h = tf.log(h / reference_h) #这四行是计算四个平移、缩放参数，注意，分母是renference_boxes的宽、高

        if scale_factors:
            t_xcenter *= scale_factors[0]
            t_ycenter *= scale_factors[1]
            t_w *= scale_factors[2]
            t_h *= scale_factors[3]

        return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w]))


def encode_boxes_rotate(unencode_boxes, reference_boxes, scale_factors=None, name='encode'):
    '''
    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
    :param reference_boxes: [H*W*num_anchors_per_location, 5]
    :return: encode_boxes [-1, 5]
    '''

    with tf.variable_scope(name):
        y_center, x_center, h, w, theta = tf.unstack(unencode_boxes, axis=1)

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

        reference_x_center = (reference_xmin + reference_xmax) / 2.
        reference_y_center = (reference_ymin + reference_ymax) / 2.
        # here maybe have logical error, reference_w and reference_h should exchange,
        # but it doesn't seem to affect the result.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin
        reference_theta = tf.ones(tf.shape(reference_xmin)) * -90

        reference_w += 1e-8
        reference_h += 1e-8
        w += 1e-8
        h += 1e-8  # to avoid NaN in division and log below

        t_xcenter = (x_center - reference_x_center) / reference_w
        t_ycenter = (y_center - reference_y_center) / reference_h
        t_w = tf.log(w / reference_w)
        t_h = tf.log(h / reference_h)
        t_theta = (theta - reference_theta) * math.pi / 180

        if scale_factors:
            t_xcenter *= scale_factors[0]
            t_ycenter *= scale_factors[1]
            t_w *= scale_factors[2]
            t_h *= scale_factors[3]
            t_theta *= scale_factors[4]

        return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w, t_theta]))