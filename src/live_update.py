import random
import math
import cv2 as cv
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure, show
from bokeh.layouts import layout, column, row


threshold = 50

vc = cv.VideoCapture(-0)
ret, color_img = vc.read()
gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
img_w = gray_img.shape[0]
img_h = gray_img.shape[1]
dnn_alpha = np.full((28, 28), 255, dtype=np.uint8)

# find white box to bound the input
c = np.argwhere(gray_img[math.floor(img_h/2), :] > threshold)
r = np.argwhere(gray_img[:, math.floor(img_w/2)] > threshold)
c1 = c.item(0)+2
c2 = c.item(len(c)-1)-2
r1 = r.item(0)+2
r2 = r.item(len(r)-1)-2

# crop out the white square
dnn_img = (255 - gray_img[r1:r2, c1:c2])
rgba_img = cv.cvtColor(color_img[r1:r2, c1:c2, :], cv.COLOR_BGR2RGBA)

dnn_img = cv.resize(dnn_img, (28, 28), interpolation=cv.INTER_CUBIC).astype('float')

min_img = np.amin(dnn_img)
max_img = np.amax(dnn_img)

dnn_img = (255*(dnn_img - min_img)/(max_img - min_img)).astype(np.uint8)
dnn_img_view = np.dstack([dnn_img, dnn_img, dnn_img, dnn_alpha])

# source = ColumnDataSource(data=dict(input_img=[rgba_img], dnn_input=[dnn_img_view]))
source = ColumnDataSource(data=dict(input_img=[], dnn_input=[]))

p1 = figure(plot_height=300, plot_width=250, title="Webcam Input", toolbar_location="below")
p1.image_rgba(image="input_img", x=0, y=0, dw=rgba_img.shape[1], dh=rgba_img.shape[0], source=source)
p1.axis.visible = False
p1.grid.visible = False
p1.x_range.range_padding = 0
p1.y_range.range_padding = 0

p2 = figure(plot_height=300, plot_width=250, title="DNN Input", toolbar_location="below")
p2.image_rgba(image="dnn_input", x=0, y=0, dw=dnn_img.shape[1], dh=dnn_img.shape[0], source=source)
p2.axis.visible = False
p2.grid.visible = False
p2.x_range.range_padding = 0
p2.y_range.range_padding = 0


def update():
    ret, color_img = vc.read()

    if ret:
        # rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        # crop out the white square
        gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
        dnn_img = (255 - gray_img[r1:r2, c1:c2])
        rgba_img = cv.cvtColor(color_img[r1:r2, c1:c2, :], cv.COLOR_BGR2RGBA)

        dnn_img = cv.resize(dnn_img, (28, 28), interpolation=cv.INTER_CUBIC).astype('float')

        dnn_img = (255 * (dnn_img - min_img) / (max_img - min_img)).astype(np.uint8)
        dnn_img_view = np.dstack([dnn_img, dnn_img, dnn_img, dnn_alpha])

        source.data = {'input_img': [np.flipud(rgba_img)], 'dnn_input' : [np.flipud(dnn_img_view)]}
    else:
        print('x')


layout = row([column([p1, p2])])

doc = curdoc()
doc.add_root(layout)
doc.add_periodic_callback(update, 200)

