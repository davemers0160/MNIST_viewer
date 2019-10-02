import os
import math
import cv2 as cv
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure, show
from bokeh.layouts import layout, column, row
import panel as pn

threshold = 128

color_img = cv.imread("D:/Projects/mnist/data/test/wc_test.png")
gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
img_h = gray_img.shape[0]
img_w = gray_img.shape[1]
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


p1 = figure(plot_height=200, plot_width=150, title="Webcam Input", toolbar_location="below")
# p1.image_rgba(image="input_img", x=0, y=0, dw=img_w, dh=img_h, source=source)
p1.image_rgba(image=[np.flipud(rgba_img)], x=0, y=0, dw=rgba_img.shape[1], dh=rgba_img.shape[0])
p1.axis.visible = False
p1.grid.visible = False
p1.x_range.range_padding = 0
p1.y_range.range_padding = 0

p2 = figure(plot_height=200, plot_width=150, title="DNN Input", toolbar_location="below")
# p2.image_rgba(image="dnn_input", x=0, y=0, dw=img_w, dh=img_h, source=source)
p2.image_rgba(image=[np.flipud(np.dstack([dnn_img, dnn_img, dnn_img, dnn_alpha]))], x=0, y=0, dw=28, dh=28)
# p2.image_rgba(image=[np.flipud(dnn_img_view)], x=0, y=0, dw=28, dh=28)
p2.axis.visible = False
p2.grid.visible = False
p2.x_range.range_padding = 0
p2.y_range.range_padding = 0


#pn.Column(pn.layout.Spacer(height=20), p1).show()

# pn.Column(pn.layout.Spacer(height=20), pn.Row(input_image, pn.layout.Spacer(width=10),
#                                   build_layer_image(ls_12, l12_data, [7, 19], 4, 1000), pn.layout.Spacer(width=10),
#                                   # build_layer_image(ls_08, l08_data, [6, 19], 2, 1000), pn.layout.Spacer(width=10),
#                                   ld01_plot(ls01_x, l01_data)), width_policy='max').show()

layout = row([column([p1, p2])])
#
show(layout)

bp = 1
