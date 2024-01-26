# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def non_maximal_suppression(boxes, scores, iou_threshold, max_output_size, name='non_maximal_suppression'):
    with tf.variable_scope(name):
        nms_index = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            name=name
        )
        return nms_index
def nms(boxes, scores, iou_threshold, max_output_size,soft_nms=False): 
    keep = []
    order = scores.argsort()[::-1]#按得分从大到小排序
    #print(order)
    num = boxes.shape[0]
    #print(num)
    suppressed = np.zeros((num), dtype=np.int)#抑制
    for _i in range(num):
        if len(keep) >= max_output_size:
            #debug()
            break
        i = order[_i]
        #print(i)
        if suppressed[i] == 1:
            continue
        keep.append(i)
        #print(keep)
	####boxes左下和右上角坐标####
        yi1=boxes[i, 0]
        xi1=boxes[i, 1]
        yi2=boxes[i, 2]
        xi2=boxes[i, 3]
        areas1=(xi2 - xi1) * (yi2 - yi1)#box1面积
        for _j in range(_i + 1, num):#start，stop
            j = order[_j]
            if suppressed[i] == 1:
                continue
            yj1=boxes[j, 0]
            xj1=boxes[j, 1]
            yj2=boxes[j, 2]
            xj2=boxes[j, 3]
            areas2=(xj2 - xj1) * (yj2 - yj1)#box2面积
            
            xx1 = np.maximum(xi1, xj1) 
            yy1 = np.maximum(yi1, yj1)
            xx2 = np.minimum(xi2, xj2)
            yy2 = np.minimum(yi2, yj2)
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            int_area=w*h#重叠区域面积
            #print(int_area)
            inter = 0.0
          
            if int_area>0:
                inter = int_area * 1.0 / (areas1 + areas2 - int_area)#IOU
                #print(inter)
            if soft_nms:
                sigma=0.5
                if inter >= iou_threshold:
                    scores[j]=np.exp(-(inter * inter)/sigma)*scores[j]
            ###nms
            else:    
                if inter >= iou_threshold:
                    suppressed[j] = 1
    order = scores.argsort()[::-1]
    #print(order)
    order = order[0:max_output_size]
    return order,scores#返回保留下来的下标
def softnms(boxes, MAX,sigma=0.5, Nt=0.7, threshold=0.001, method=2):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

# boxes = np.array([[100, 100, 150, 168, 0.63], [166, 70, 312, 190, 0.55], [
                #  221, 250, 389, 500, 0.79], [12, 190, 300, 399, 0.9], [28, 130, 134, 302, 0.3]])
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
    # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

    # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

    # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
    # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) *
                               (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                    #print(boxes[:, 4])

            # if box score falls below threshold, discard the box by swapping with last box
            # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N-1, 0]
                        boxes[pos, 1] = boxes[N-1, 1]
                        boxes[pos, 2] = boxes[N-1, 2]
                        boxes[pos, 3] = boxes[N-1, 3]
                        boxes[pos, 4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1
    keep = [i for i in range(MAX)]
    return keep
def soft_nms(dets, sc, MAX,Nt=0.3, sigma=0.5, thresh=0.001, method=2):
  
    keep = tf.py_func(py_cpu_softnms,
                          inp=[dets, sc, MAX,Nt, sigma, thresh, method],
                          Tout=tf.int64)
    return keep
def py_cpu_softnms(dets, sc, MAX,Nt, sigma, thresh, method):
    

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    #N =2000
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        #print(dets[i,:])
        tBD = dets[i, :].copy()
        #print(tBD)
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4]
    keep = inds.astype(int)
    #print(dets)
    keep = keep[0:MAX]
    #print(keep)
    return np.array(keep, np.int64)