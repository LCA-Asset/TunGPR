# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
from data.io import image_preprocess
from IPython.core import debugger
debug = debugger.Pdb().set_trace
from libs.box_utils.coordinate_convert import forward_convert2
from libs.configs import cfgs
def read_single_example_and_decode(filename_queue):


    # tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    # reader = tf.TFRecordReader(options=tfrecord_options)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)

    img = tf.reshape(img, shape=[img_height, img_width, 3])

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.float32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])

    num_objects = tf.cast(features['num_objects'], tf.int32)
    return img_name, img, gtboxes_and_label, num_objects

def read_and_prepocess_single_img(filename_queue, shortside_len, is_training):

    img_name, img, gtboxes_and_label, num_objects = read_single_example_and_decode(filename_queue)
    # img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    #img = img/255.0
    #img = (img - tf.constant([103.939, 116.779, 123.68]))/255.0
    img = (img - tf.constant([116.569, 234.389, 132.441]))/tf.constant([45.702, 59.647, 47.026])#224de��ֵ��׼��
    #img = (img - tf.constant([116.5, 234.4, 132.2]))/tf.constant([46.7, 60.9, 48.2])#533de��ֵ��׼��
    if is_training:
        #img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                       # target_shortside_len=shortside_len)
        gtboxes_and_label = tf.py_func(forward_convert2,
                                             inp=[gtboxes_and_label],
                                             Tout=tf.float32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])
        #img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img, gtboxes_and_label=gtboxes_and_label)

    else:
        gtboxes_and_label = tf.py_func(forward_convert2,
                                             inp=[gtboxes_and_label],
                                             Tout=tf.float32)
      
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])
        #img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                  #  target_shortside_len=shortside_len)

    return img_name, img, gtboxes_and_label, num_objects


def next_batch(dataset_name, batch_size, shortside_len, is_training,is_valid):
    if dataset_name not in ['ship', 'spacenet', 'hyperbola', 'coco',
                            'gprmax', 'newgprmax1', 'newgprmax2', 'newgprmax3',
                            'aligned_hyperbola', 'r_aligned_hyperbola']:
        raise ValueError('dataSet name must be in pascal or coco')

    if is_training:
        pattern = os.path.join('../data/tfrecords', dataset_name + '_train*')
    elif is_valid:
        pattern = os.path.join('../data/tfrecords', dataset_name + '_valid*')
    else:
        pattern = os.path.join('../data/tfrecords', dataset_name + '_test*')

    print('tfrecord path is -->', os.path.abspath(pattern))
    filename_tensorlist = tf.io.match_filenames_once(pattern)
    print('file:', filename_tensorlist)
    filename_queue = tf.train.string_input_producer(filename_tensorlist)
    img_name, img, gtboxes_and_label, num_obs = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                              is_training=is_training)
    img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch = \
        tf.train.batch(
                       [img_name, img, gtboxes_and_label, num_obs],
                       batch_size=batch_size,
                       capacity=100,
                       num_threads=16,
                       dynamic_pad=True)
    return img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
        next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                   batch_size=cfgs.BATCH_SIZE,
                   shortside_len=cfgs.SHORT_SIDE_LEN,
                   is_training=True,
                   is_valid=False)
    gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 9])

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        img_name_batch_, img_batch_, gtboxes_and_label_batch_, num_objects_batch_\
            = sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch])

        coord.request_stop()
        coord.join(threads)