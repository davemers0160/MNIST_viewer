import os
import math
import cv2 as cv
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure, show
from bokeh.layouts import layout, column, row
import panel as pn


script_path = os.path.realpath(__file__)

# set up some global variables that will be used throughout the code
# read only
threshold = 80
image_name = os.path.dirname(os.path.dirname(script_path)) + "/input_test.png"


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_squares(img, threshold):
    squares = []
    # img = cv.GaussianBlur(img, (5, 5), 0)

    _retval, bin = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    bin = cv.Canny(bin, 0, 50, apertureSize=5, L2gradient=False)
    bin = cv.dilate(bin, None)

    # cv.imshow('frame', bin)
    # cv.waitKey(1)

    contours, _hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            if max_cos < 0.1:
                squares.append(cnt)

    return squares


color_img = cv.imread(image_name)
gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
img_h = gray_img.shape[0]
img_w = gray_img.shape[1]
dnn_alpha = np.full((28, 28), 255, dtype=np.uint8)

squares = find_squares(gray_img, threshold)

x1 = np.amin(squares[0][:, 0]) + 2
x2 = np.amax(squares[0][:, 0]) - 2
y1 = np.amin(squares[0][:, 1]) + 2
y2 = np.amax(squares[0][:, 1]) - 2

cv.drawContours(color_img, squares, -1, (0, 255, 0), 1)
cv.imshow('squares', color_img)
ch = cv.waitKey(1)

dnn_img = (255 - gray_img[y1:y2, x1:x2])
cv.imshow('crop', dnn_img)
ch = cv.waitKey()

bp = 1

# find white box to bound the input
# c = np.argwhere(gray_img[math.floor(img_h/2), :] > threshold)
# r = np.argwhere(gray_img[:, math.floor(img_w/2)] > threshold)
# c1 = c.item(0)+2
# c2 = c.item(len(c)-1)-2
# r1 = r.item(0)+2
# r2 = r.item(len(r)-1)-2

# crop out the white square
# dnn_img = (255 - gray_img[r1:r2, c1:c2])
# rgba_img = cv.cvtColor(color_img[r1:r2, c1:c2, :], cv.COLOR_BGR2RGBA)
#
# dnn_img = cv.resize(dnn_img, (28, 28), interpolation=cv.INTER_CUBIC).astype('float')

min_img = np.amin(dnn_img)
max_img = np.amax(dnn_img)

dnn_img = (255*(dnn_img - min_img)/(max_img - min_img)).astype(np.uint8)
# dnn_img_view = np.dstack([dnn_img, dnn_img, dnn_img, dnn_alpha])


# p1 = figure(plot_height=200, plot_width=150, title="Webcam Input", toolbar_location="below")
# # p1.image_rgba(image="input_img", x=0, y=0, dw=img_w, dh=img_h, source=source)
# p1.image_rgba(image=[np.flipud(rgba_img)], x=0, y=0, dw=rgba_img.shape[1], dh=rgba_img.shape[0])
# p1.axis.visible = False
# p1.grid.visible = False
# p1.x_range.range_padding = 0
# p1.y_range.range_padding = 0

# p2 = figure(plot_height=200, plot_width=150, title="DNN Input", toolbar_location="below")
# # p2.image_rgba(image="dnn_input", x=0, y=0, dw=img_w, dh=img_h, source=source)
# p2.image_rgba(image=[np.flipud(np.dstack([dnn_img, dnn_img, dnn_img, dnn_alpha]))], x=0, y=0, dw=28, dh=28)
# # p2.image_rgba(image=[np.flipud(dnn_img_view)], x=0, y=0, dw=28, dh=28)
# p2.axis.visible = False
# p2.grid.visible = False
# p2.x_range.range_padding = 0
# p2.y_range.range_padding = 0



# layout = row([column([p1, p2])])
# #
# show(layout)

bp = 1
