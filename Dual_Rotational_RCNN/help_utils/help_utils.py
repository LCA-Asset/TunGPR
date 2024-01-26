# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
from libs.label_name_dict.label_dict import LABEl_NAME_MAP
from IPython.core import debugger
debug = debugger.Pdb().set_trace

def show_boxes_in_img(img, boxes_and_label):
    '''

    :param img:
    :param boxes: must be int
    :return:
    '''
    boxes_and_label = boxes_and_label.astype(np.int64)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
    for box in boxes_and_label:
        ymin, xmin, ymax, xmax, label = box[0], box[1], box[2], box[3], box[4]

        category = LABEl_NAME_MAP[label]

        color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        cv2.rectangle(img,
                      pt1=(xmin, ymin),
                      pt2=(xmax, ymax),
                      color=color)
        cv2.putText(img,
                    text=category,
                    org=((xmin+xmax)//2, (ymin+ymax)//2),
                    fontFace=1,
                    fontScale=1,
                    color=(0, 0, 255))

    cv2.imshow('img_', img)
    cv2.waitKey(0)


def draw_box_cv(img_name, boxes, labels, scores, is_valid):
    if is_valid:
        mode = 'valid'
    else:
        mode = 'test'
    img = cv2.imread('C:\\Users\\zhuhu\\Desktop\\R2DCNN\\R2CNN_FPN_Tensorflow-master\\data\\%s\\JPEGImages\\%s'%(mode, img_name))

        #img * np.array([45.702, 59.647, 47.026])
    #print(img)
    # img = abs(img)*np.array([100, 100, 100])
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    # img = np.array(img*255/np.max(img), np.uint8)

    num_of_object = 0
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        label = labels[i]
        ll = 0
        #print(label)
        if label != 0:
            num_of_object += 1
            if label == 1:
               color = (0, 255,0)
               ll = 180
            elif label == 2:
               color = (0, 128,255)
               ll = 180
            elif label == 3:
               color = (0, 255,255)
               ll = 160
            elif label == 4:
               color = (255, 0,255)
               ll = 260
            elif label == 5:
               color = (203, 192,255)
               ll = 220
            elif label == 6:
               color = (255, 255,0)
               ll = 200
            elif label == 7:
               color = (13,23,227)
               ll = 300
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

            category = LABEl_NAME_MAP[label]

            if scores is not None:
                cv2.rectangle(img,
                              pt1=(xmin, ymin-30),
                              pt2=(xmin+ll, ymin),
                              color=color,
                              thickness=-1)
                cv2.putText(img,
                            text=category+":"+'{:.2f}'.format(scores[i]),
                            org=(xmin+5, ymin-5),
                            fontFace=2,
                            fontScale=1,
                            thickness=2,
                            color=(0, 0, 0))
            else:
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin + 40, ymin + 15),
                              color=color,
                              thickness=-1)
                cv2.putText(img,
                            text=category,
                            org=(xmin, ymin + 10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
    #cv2.putText(img,
    #            text=str(num_of_object),
    #            org=((img.shape[1]) // 2, (img.shape[0]) // 2),
    #            fontFace=3,
    #            fontScale=1,
    #            color=(255, 0, 0))
    return img


#def draw_rotate_box_cv(img_name_batch, img, boxes, hboxes, labels, scores):
#    
#    img = img * np.array([45.702, 59.647, 47.026])
#    #print(img)
#    img = img + np.array([116.569, 234.389, 132.441])
#    boxes = boxes.astype(np.int64)
#    hboxes = hboxes.astype(np.int64)
#    labels = labels.astype(np.int32)
#    img = np.array(img, np.float32)
#    img = np.array(img*255/np.max(img), np.uint8)
#
#    num_of_object = 0
#    for i, box in enumerate(boxes):
#        y_c, x_c, h, w, theta = box[0], box[1], box[2], box[3], box[4]
#        hbox = hboxes[i]
#        y1, x1, y2, x2 = hbox[0], hbox[1], hbox[2], hbox[3]
#        label = labels[i]
#        ll = 0
#        #print(label)
#        if label != 0:
#            num_of_object += 1
#            if label == 1:
#               color = (0, 255,0)
#               ll = 180+110
#            elif label == 2:
#               color = (0, 128,255)
#               ll = 180+110
#            elif label == 3:
#               color = (0, 255,255)
#               ll = 260
#            elif label == 4:
#               color = (255, 0,255)
#               ll = 420
#            elif label == 5:
#               color = (203, 192,255)
#               ll = 352
#            elif label == 6:
#               color = (255, 255,0)
#               ll = 320
#            elif label == 7:
#               color = (13,23,227)
#               ll = 480
#            rect = ((x_c, y_c), (w, h), theta)
#            rect = cv2.boxPoints(rect)
#            rect = np.int0(rect)
#            cv2.drawContours(img, [rect], -1, color, 3)
#
#            category = LABEl_NAME_MAP[label]
#            if img_name_batch == b'000052.jpg':
#               if label ==4:
#                   x1 = x1-70
#            if img_name_batch == b'000118.jpg':
#               y1=y1+20
#            if img_name_batch == b'000490.jpg':
#               if label ==7:
#                  x1=x1-150
#               if label ==5:
#                  x1=x1-120
#            if img_name_batch == b'007035.jpg':      
#               x1=x1-70 
#            if img_name_batch == b'007004.jpg':    
#               if label ==5:
#                   y1=y1+10
#            if scores is not None:
#                cv2.rectangle(img,
#                              pt1=(x1-10, y1-60),
#                              pt2=(x1 + ll, y1),
#                              color=color,
#                              thickness=-1)
#                cv2.putText(img,
#                            text=category+":"+'{:.2f}'.format(scores[i]),
#                            org=(x1, y1-15),
#                            fontFace=4,
#                            fontScale=1.5,
#                            thickness=4,
#                            color=(0,0,0))
#            else:
#                cv2.rectangle(img,
#                              pt1=(x1, y1-30),
#                              pt2=(x1 + ll-80, y1),
#                              color=color,
#                              thickness=-1)
#                cv2.putText(img,
#                            text=category,
#                            org=(x1+5, y1-5),
#                            fontFace=2,
#                            fontScale=1,
#                            thickness=2,
#                            color=(0, 0, 0))
#              
#
#    return img
def draw_rotate_box_cv(img_name_batch,img, boxes, hboxes, labels, scores):
    img = img * np.array([45.702, 59.647, 47.026])
    #print(img)
    img = img + np.array([116.569, 234.389, 132.441])
    boxes = boxes.astype(np.int64)   
    hboxes = hboxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
    num_of_object = 0
    mask = np.zeros(img.shape)
    img2 = np.array(img)
    img3 = np.array(img)
    rect_gather = []
#    if img_name_batch == b'000335.jpg': 
#       boxes = np.array([boxes[1],boxes[2],boxes[0]])
#       hboxes = np.array([hboxes[1],hboxes[2],hboxes[0]])
#       labels = np.array([labels[1],labels[2],labels[0]])             
    for i, box in enumerate(boxes):
        
        y_c, x_c, h, w, theta = box[0], box[1], box[2], box[3], box[4]
        hbox = hboxes[i]
        y1, x1, y2, x2 = hbox[0], hbox[1], hbox[2], hbox[3]
        label = labels[i]
        ll = 0
        #print(label)
        #if label != 0:
        #    num_of_object += 1
        #    if label == 1:
        #       color = (0, 255,0)
        #       ll = 180+120
        #    elif label == 2:
        #       color = (0, 128,255)
        #       ll = 180+130
        #    elif label == 3:
        #       color = (0, 255,255)
        #       ll = 270
        #    elif label == 4:
        #       color = (255, 0,255)
        #       ll = 450
        #    elif label == 5:
        #       color = (203, 192,255)
        #       ll = 380
        #    elif label == 6:
        #       color = (255, 255,0)
        #       ll = 340
        #    elif label == 7:
        #       color = (13,23,227)
        #       ll = 520
        if label != 0:
            num_of_object += 1
            if label == 1:
               color = (255, 0,0)
               ll = 180+110
            elif label == 2:
               color = (0, 0,255)
               ll = 180+110
            elif label == 3:
               color = (0, 255,255)
               ll = 260
            elif label == 4:
               color = (255, 0,255)
               ll = 420
            elif label == 5:
               color = (203, 192,255)
               ll = 352
            elif label == 6:
               color = (255, 255,0)
               ll = 320
            elif label == 7:
               color = (13,23,227)
               ll = 480
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect) #这里得到了倾斜矩形的四个顶点坐标
            rect = np.int0(rect)
            rect_gather.append(rect) #顶点坐标另存为并输出.npy文件
            cv2.drawContours(img2, [rect], -1, color, 3)
            cv2.rectangle(img3, (x1, y1), (x2, y2), color, 3)
            cv2.drawContours(mask, [rect], -1, (1, 1, 1), -1)
            
    roi = img*mask
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if all(roi[i][j] == [0, 0, 0]):
                roi[i][j] = [255, 255, 255]
    roi = roi.astype(np.float32)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return img2, img3, roi_gray, mask, rect_gather


def print_tensors(tensor, tensor_name):

    def np_print(ary):
        ary = ary + np.zeros_like(ary)
        print(tensor_name + ':', ary)

        print('shape is: ',ary.shape)
        print(10*"%%%%%")
        return ary
    result = tf.py_func(np_print,
                        [tensor],
                        [tensor.dtype])
    result = tf.reshape(result, tf.shape(tensor))
    result = tf.cast(result, tf.float32)
    sum_ = tf.reduce_sum(result)
    tf.summary.scalar('print_s/{}'.format(tensor_name), sum_)
