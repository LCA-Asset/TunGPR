# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
from libs.configs import cfgs
import numpy as np
import tensorflow as tf
import glob
import cv2
from libs.label_name_dict.label_dict import *
from help_utils.tools import *
from IPython.core import debugger
debug = debugger.Pdb().set_trace
tf.app.flags.DEFINE_string('VOC_dir', None, 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotation', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'JPEGImages', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', cfgs.ROOT_PATH + '/data/tfrecords/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'pascal', 'dataset')
FLAGS = tf.app.flags.FLAGS

"""
Command example:
python convert_data_to_tfrecord.py 
--VOC_dir=C:/Users/zhuhu/Desktop/Manuscript/Dataset/Synthetic/Syn_single_hyperbola/train/ 
--xml_dir=Annotation --image_dir=JPEGImages --save_name=train --img_format=.jpg --dataset=syn_alig
"""


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8') if type(value)==str else value]))


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            print(NAME_LABEL_MAP)
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(float(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.float32)
    print(gtbox_label)
    return img_height, img_width, gtbox_label


def convert_ro_to_aligned(ro_xml_path, aligned_template, save_path):
    """
    This is to convert roLabelImg files to axis-aligned label files with
    a maximum external rectangle.
    """

    tree = ET.parse(ro_xml_path)
    root = tree.getroot()
    position = {'left': [], 'right': []}
    coordinates = {'x': [], 'y': []}
    flag = ''
    for child_of_root in root:
        if child_of_root.tag == 'path':
            path = child_of_root.text
        if child_of_root.tag == 'object':
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    flag = child_item.text.split('_')[0]
                if child_item.tag == 'robndbox':
                    for node in child_item:
                        position[flag].append(float(node.text))

    for key in position.keys():
        cx, cy, w, h, angle = [position[key][i] for i in range(5)]
        cross_line = math.sqrt(w**2+h**2)
        if key == 'left':
            beta_12 = math.pi - angle - math.atan(h/w)
            beta_34 = math.pi - angle + math.atan(h/w)
            x_l = (cross_line) / 2 * math.cos(beta_12)
            y_l = (cross_line) / 2 * math.sin(beta_12)
            x1, y1, x2, y2 = cx - x_l, cy + y_l, cx + x_l, cy - y_l
            x_l = (cross_line) / 2 * math.cos(beta_34)
            y_l = (cross_line) / 2 * math.sin(beta_34)
            x3, y3, x4, y4 = cx - x_l, cy + y_l, cx + x_l, cy - y_l
            for i in [x1, x2, x3, x4]:
                coordinates['x'].append(i)
            for i in [y1, y2, y3, y4]:
                coordinates['y'].append(i)
        if key == 'right':
            beta_12 = angle - math.atan(h / w)
            beta_34 = angle + math.atan(h / w)
            x_l = (cross_line) / 2 * math.cos(beta_12)
            y_l = (cross_line) / 2 * math.sin(beta_12)
            x1, y1, x2, y2 = cx - x_l, cy - y_l, cx + x_l, cy + y_l
            x_l = (cross_line) / 2 * math.cos(beta_34)
            y_l = (cross_line) / 2 * math.sin(beta_34)
            x3, y3, x4, y4 = cx - x_l, cy - y_l, cx + x_l, cy + y_l
            for i in [x1, x2, x3, x4]:
                coordinates['x'].append(i)
            for i in [y1, y2, y3, y4]:
                coordinates['y'].append(i)
            xmin, xmax, ymin, ymax = min(coordinates['x']), max(coordinates['x']), \
                                     min(coordinates['y']), max(coordinates['y'])
            values = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
    tree = ET.parse(aligned_template)
    root = tree.getroot()
    for child_of_root in root:
        if child_of_root.tag == 'path':
            child_of_root.text == path
        if child_of_root.tag == 'object':
            for child_item in child_of_root:
                if child_item.tag == 'bndbox':
                    for node in child_item:
                        node.text = str(int(values[node.tag]))
    tree.write(os.path.join(save_path))


def convert_pascal_to_tfrecord():
    xml_path = FLAGS.VOC_dir + FLAGS.xml_dir
    image_path = FLAGS.VOC_dir + FLAGS.image_dir
    save_path = FLAGS.save_dir + FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)
    for count, xml in enumerate(glob.glob(xml_path + '/*.xml')):
        # to avoid path error in different development platform
        xml = xml.replace('\\', '/')

        img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name
        print(img_name)
        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
        #debug()
        print(gtbox_label)
        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            #'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(xml_path + '/*.xml')))

    print('\nConversion is complete!')


if __name__ == '__main__':
    # xml_path = r'C:\Users\zhuhu\Desktop\0_2_7_20.xml'
    # img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()

    # path = r'C:\Users\zhuhu\Desktop\Manuscript\Dataset\Real\test\Annotation'
    # for i in os.listdir(path):
    #     convert_ro_to_aligned(os.path.join(path, i),
    #                           r'C:\Users\zhuhu\Desktop\0_2_6_20.xml',
    #                           os.path.join(r'C:\Users\zhuhu\Desktop\Manuscript\Dataset\Real\test\Annotations', i))
