import numpy as np
from PIL import Image
from pylab import plot
import cv2 as cv
import os
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import random
import math
from scipy.optimize import minimize

box_position = '4_77.npy' # This is the box positions output from the Dual Rotational CNN
raw_img = 'raw_4_77.jpg'
bi_img = 'bi_4_77.jpg' # The mask can be obtained by applying the OTSU threshold method within the bounding boxes.

position = np.load(box_position)
raw = cv.imread(raw_img)
img = cv.imread(bi_img)
binary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
binary = np.rint(binary/255)

n_of_hyperbola = int(position.shape[0]/2)
for n in range(n_of_hyperbola):
    removed = np.zeros(binary.shape)
    # Search for the peak
    l1, l2, l3, l4 = position[n][:]
    distance = 512
    box = 1
    for j in range(position.shape[0]-n_of_hyperbola):
        r1, r2, r3, r4 = position[j+n_of_hyperbola][:]
        cal_distance = np.sqrt(np.sum(np.power(l3-r3, 2)))
        if cal_distance<distance:
            distance = cal_distance
            box = j
    r1, r2, r3, r4 = position[n_of_hyperbola +box]
    l_xc = (l1[0]+l3[0])/2
    l_yc = (l1[1]+l3[1])/2
    r_xc = (r1[0]+r3[0])/2
    r_yc = (r1[1]+r3[1])/2
    point_1 = np.array([l_xc, l_yc])
    point_2 = np.array([r_xc, r_yc])
    point_3 = np.array(l4)
    point_4 = np.array(r2)
    x = np.array([l_xc, r2[0], l4[0], r_xc])
    y_para = np.array([l_yc, r2[1], l4[1], r_yc])
    para = np.polyfit(x, y_para, 2)
    eq = np.poly1d(para)
    x_new = np.array([l_xc, r2[0], (r2[0]+l4[0])/2, l4[0], r_xc])
    y_fitted = eq(x_new)

    # plt.imshow(img)
    # plt.plot(x, y,'r', label = 'Original curve')
    # plt.plot(x_new, y_fitted, '-b', label ='Fitted curve')
    # plt.legend()
    # Search from the column of l2 towards the right until the column of r4
    start_column = l1[0]
    end_column = r1[0]
    start_row = np.minimum(l3[1], r3[1])
    end_row = np.maximum(l1[1], r1[1])
    gather = []
    column_gather = []
    for column in range(start_column, end_column+1):
        cluster = []
        number = 0
        for row in range(start_row+1, end_row+1):
            if binary[row][column] == 0.0: # Please note that slicing with Numpy starts with rows and then columns
                continue
            if binary[row][column] == 1.0:
                if binary[row-1][column] == 0.0 or binary[row+1][column] == 0.0:
                    number += 1
                    cluster.append([column, row])
                    if number>0 and (number%2)== 0:
                        column_gather.append(cluster)
                        cluster = []
                else:
                    continue

        if column == end_column:
            gather.append(column_gather)
    to_be_removed = []
    for i in range(len(column_gather)):
        column = column_gather[i][0][0]
        start_row = column_gather[i][0][1]
        end_row = column_gather[i][1][1]
        mid_row = (start_row+end_row)/2.0
        y = eq(column)
        if abs(mid_row-y)>15:
            to_be_removed.append(i)
    to_be_removed.sort(reverse=True)
    for i in to_be_removed:
        column_gather.pop(i)
    for i in range(len(column_gather)):
        column = column_gather[i][0][0]
        start_row = column_gather[i][0][1]
        end_row = column_gather[i][1][1]
        for j in range(start_row, end_row+1):
            removed[j][column] = 1
    point_x = []
    point_y = []
    for i in range(len(column_gather)):
        point_x.append(column_gather[i][0][0])
        point_y.append((column_gather[i][0][1] + column_gather[i][1][1]) / 2)
    #plt.imshow(img)
    #plt.plot(point_x, point_y, 'r', label='Original point')

    def func(x, h, k, a, b):
        return np.sqrt(np.square(a * b) + np.square(b * (x - h))) + k
    popt, pcov = curve_fit(func, point_x, point_y)
    y_hyper = func(point_x, popt[0], popt[1], popt[2], popt[3])
    plt.imshow(raw)
    plt.axis('off')
    legend_font = {'family': 'Times New Roman',
                   'size': 40,
                   }
    if n == 0:
        plt.plot(point_x, y_hyper, color= 'red', label='Fitted', linestyle='--', linewidth=3)
        legend = plt.legend(loc='upper left', frameon=False, prop=legend_font)
        plt.setp(legend.get_texts(), color=(0,0,0,0))


    else:
        plt.plot(point_x, y_hyper, color='red', linestyle='--', linewidth=3)
        plt.legend(labelcolor='white', loc='upper left', frameon=False, prop=legend_font)
    cv.imwrite('%s_removed.jpg'%box_position.split('.')[0], removed*255)

# Plot hyperbola points in binary image
    # bi = binary*255
    # stacked_bi = np.stack((bi,)*3, axis=-1)
    # for i in range(len(point_x)):
    #     center = (int(point_x[i]), int(point_y[i]))
    #     cv.circle(stacked_bi, center, radius=1, color=(0, 0, 255))
    # cv.imwrite('points_%s.jpg'%number, stacked_bi)
#plt.plot(x, y_para, 'o', color='b', label='Control points')