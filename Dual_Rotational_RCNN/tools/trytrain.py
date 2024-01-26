# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')

import tensorflow.contrib.slim as slim
import time
from data.io.read_tfrecord import next_batch
from libs.networks.network_factory import get_flags_byname
from libs.networks.network_factory import get_network_byname
from libs.configs import cfgs
from libs.rpn import build_rpn
from libs.fast_rcnn import build_fast_rcnn
from help_utils.tools import *
from libs.box_utils.show_box_in_tensor import *
from tools import restore_model
from libs.box_utils.coordinate_convert import back_forward_convert
from libs.box_utils.boxes_utils import get_horizen_minAreaRectangle


RESTORE_FROM_RPN = False
FLAGS = get_flags_byname(cfgs.NET_NAME)
os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


def train():
    with tf.Graph().as_default():
        with tf.name_scope('get_batch'):
            img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
                next_batch(dataset_name=cfgs.DATASET_NAME,
                           batch_size=cfgs.BATCH_SIZE,
                           shortside_len=cfgs.SHORT_SIDE_LEN,
                           is_training=True,
                           is_valid=False)
            gtboxes_and_label = tf.py_func(back_forward_convert,
                                           inp=[tf.squeeze(gtboxes_and_label_batch, 0)],
                                           Tout=tf.float32)
            gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 6])

            gtboxes_and_label_minAreaRectangle = get_horizen_minAreaRectangle(gtboxes_and_label)

            gtboxes_and_label_minAreaRectangle = tf.reshape(gtboxes_and_label_minAreaRectangle, [-1, 5])

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        _gtboxes_and_label = sess.run(gtboxes_and_label_minAreaRectangle)
        print('start------------------------------------')
        print('_gtboxes:', str(_gtboxes_and_label))

train()