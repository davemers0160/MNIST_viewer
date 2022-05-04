#
#
# To run this file: bokeh serve --show mnist_viewer.py
#

import platform
import os
import math
#import ctypes as ct
import setuptools
from cffi import FFI
import numpy as np
import cv2 as cv
import bokeh
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Div
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, Spacer
from find_squares import find_squares
from bokeh.server.server import Server

script_path = os.path.realpath(__file__)

# set up some global variables that will be used throughout the code
# read only
threshold = 150
dnn_alpha = np.full((28, 28), 255, dtype=np.uint8)
update_time = 200
ffi = FFI()
plot_width = 780


use_webcam = False
if use_webcam:
    vc = cv.VideoCapture(0)
else:
    image_name = os.path.dirname(os.path.dirname(script_path)) + "/input_test.png"
    print("image path: " + str(image_name))

# modify these to point to the right locations
if platform.system() == "Windows":
    libname = "mnist_lib.dll"
    home = script_path[0:2]         # assumes that the viewer project is placed into the same folder as the dll project
    lib_location = home + "/Projects/mnist_net_lib/build/Release/" + libname
    weights_file = home + "/Projects/mnist_net_lib/nets/mnist_net_pso_14_97.dat"
elif platform.system() == "Linux":
    libname = "libmnist_lib.so"
    home = os.path.expanduser('~')
    lib_location = home + "/Projects/mnist_net_lib/build/" + libname
    weights_file = home + "/Projects/mnist_net_lib/nets/mnist_net_pso_14_97.dat"
else:
    quit()

# read and write global
mnist_lib = ffi.dlopen(lib_location)
x_r = y_r = min_img = max_img = 0

ffi.cdef('''
struct layer_struct{
    unsigned int k;
    unsigned int n;
    unsigned int nr;
    unsigned int nc;
    unsigned int size;
};

void init_net(const char *net_name);
void run_net(unsigned char* input, unsigned int nr, unsigned int nc, unsigned int *res);
void get_layer_01(struct layer_struct *data, const float** data_params);
void get_layer_02(struct layer_struct *data, const float** data_params);
void get_layer_08(struct layer_struct *data, const float** data_params);
void get_layer_12(struct layer_struct *data, const float** data_params);
''')

arch_div = Div(text="""<img src="http://mnist_architecture.png", width="900" height="260">""", width=900, height=260)

#<p style="margin-top:10px; margin-bottom:0px; font-size:20px"><B> Result</B></p>
result_div = Div(text="""<p style="margin-top:8px; font-size:180px;">0</p>""", width=50, height=260)

source = ColumnDataSource(data=dict(input_img=[], dnn_input=[], l12_img=[], l08_img=[]))

l02 = figure(plot_height=350, plot_width=plot_width, title="Layer 02 Output")
l01 = figure(plot_height=350, plot_width=plot_width, title="Layer 01 Output")

# class layer_struct(ct.Structure):
#     _fields_ = [("k", ct.c_uint), ("n", ct.c_uint), ("nr", ct.c_uint), ("nc", ct.c_uint), ("size", ct.c_uint)]


def jet_clamp(v):
    v[v < 0] = 0
    v[v > 1] = 1
    return v


def jet_colormap(n):
    t_max = n+100
    t_min = 100

    t_range = t_max - 0
    t_avg = (t_max + 0) / 2.0
    t_m = (t_max - t_avg) / 2.0

    t = np.arange(0, t_max)

    rgb = np.empty((t_max, 3), dtype=np.uint8)
    rgb[:, 0] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg - t_m)))).astype(np.uint8)
    rgb[:, 1] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg)))).astype(np.uint8)
    rgb[:, 2] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg + t_m)))).astype(np.uint8)

    cm = ['#000000']
    for z in rgb:
        cm.append(("#" + ("{:0>2x}" * len(z))).format(*z))

    return cm


def jet_color(t, t_min, t_max):

    t_range = t_max - t_min
    t_avg = (t_max + t_min) / 2.0
    t_m = (t_max - t_avg) / 2.0

    rgb = np.empty((t.shape[0], t.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg - t_m)))).astype(np.uint8)
    rgb[:, :, 1] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg)))).astype(np.uint8)
    rgb[:, :, 2] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg + t_m)))).astype(np.uint8)

    return rgb


def build_layer_image(ls, ld, cell_dim, padding, map_length):

    min_v = np.amin(ld)
    max_v = np.amax(ld)
    img_array = np.floor((map_length)*(ld - min_v)/(max_v - min_v)) + 100

    img_length = ls.nr * ls.nc

    img_h = (ls.nr + padding)*(cell_dim[0]-1) + ls.nr + 2*padding
    img_w = (ls.nc + padding)*(cell_dim[1]-1) + ls.nc + 2*padding
    layer_img = np.zeros((img_h, img_w), dtype=np.float)

    r = padding
    c = padding

    for idx in range(ls.k):
        p1 = (idx * img_length)
        p2 = ((idx+1) * img_length)

        layer_img[r:r+ls.nr, c:c+ls.nc] = np.reshape(img_array[p1:p2], [ls.nr, ls.nc])

        c = c + (ls.nc + padding)
        if(c >= img_w):
            c = padding
            r = r + (ls.nr + padding)

    return layer_img


def init_mnist_lib():
    global mnist_lib, x_r, y_r, min_img, max_img, ls_12, ld_12, ls_08, ld_08, ls_02, ld_02, ls_01, ld_01, res

    # initialize the network with the weights file
    mnist_lib.init_net(weights_file.encode('utf-8'))

    # load in an image and convert to grayscale
    if use_webcam:
        ret, color_img = vc.read()
    else:
        color_img = cv.imread(image_name)

    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    img_h = gray_img.shape[0]
    img_w = gray_img.shape[1]

    # find the outer square <- make sure that the camera only sees a border of black and the white square
    squares = find_squares(gray_img, threshold)

    # if we found a square then get the cropping ranges
    if len(squares) > 0:
        xm = squares[0][:, 0] < (img_w/2)
        ym = squares[0][:, 1] < (img_h/2)
        x_r = slice(np.amax(squares[0][xm, 0]) + 5, np.amin(squares[0][np.logical_not(xm), 0]) - 5)
        y_r = slice(np.amax(squares[0][ym, 1]) + 5, np.amin(squares[0][np.logical_not(ym), 1]) - 5)

    else:
        x_r = slice(0, img_w)
        y_r = slice(0, img_h)

    dnn_img = (255 - gray_img[y_r, x_r])
    dnn_img = cv.resize(dnn_img, (28, 28), interpolation=cv.INTER_CUBIC).astype('float')

    min_img = np.amin(dnn_img)
    max_img = np.amax(dnn_img)

    # instantiate the run_net variables
    res = ffi.new('unsigned int *')

    # instantiate the get_layer_12 variables
    ls_12 = ffi.new('struct layer_struct*')
    ld_12 = ffi.new('float**')

    # instantiate the get_layer_08 variables
    ls_08 = ffi.new('struct layer_struct*')
    ld_08 = ffi.new('float**')

    # instantiate the get_layer_02 variables
    ls_02 = ffi.new('struct layer_struct*')
    ld_02 = ffi.new('float**')

    # instantiate the get_layer_01 variables
    ls_01 = ffi.new('struct layer_struct*')
    ld_01 = ffi.new('float**')


def update():
    global mnist_lib, x_r, y_r, min_img, max_img, l01, l02, ls_12, ld_12, ls_08, ld_08, ls_02, ld_02, ls_01, ld_01, res

    # load in an image and convert to grayscale
    if use_webcam:
        ret, color_img = vc.read()
    else:
        color_img = cv.imread(image_name)

    try:
        gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    except cv.error as e:
        # print('wait: ' + str(e))
        return

    # crop out the white square
    dnn_img = (255 - gray_img[y_r, x_r])
    rgba_img = cv.cvtColor(color_img[y_r, x_r, :], cv.COLOR_BGR2RGBA)

    dnn_img = cv.resize(dnn_img, (28, 28), interpolation=cv.INTER_CUBIC).astype('float')
    dnn_img = np.minimum(np.maximum((255 * (dnn_img - min_img)) / abs(max_img - min_img), 0), 255).astype(np.uint8)
    dnn_img_view = np.dstack([dnn_img, dnn_img, dnn_img, dnn_alpha])

    # run the image on network and get the results
    mnist_lib.run_net(dnn_img.tobytes(), dnn_img.shape[0], dnn_img.shape[1], res)

    # get the Layer 12 data
    mnist_lib.get_layer_12(ls_12, ld_12)
    l12_data = np.frombuffer(ffi.buffer(ld_12[0], ls_12.size * 4), dtype=np.float32)

    # get the Layer 08 data
    mnist_lib.get_layer_08(ls_08, ld_08)
    l08_data = np.frombuffer(ffi.buffer(ld_08[0], ls_08.size * 4), dtype=np.float32)

    # get the Layer 02 data
    mnist_lib.get_layer_02(ls_02, ld_02)
    l02_data = np.frombuffer(ffi.buffer(ld_02[0], ls_02.size * 4), dtype=np.float32)
    ls02_x = np.arange(0, ls_02.k, 1)

    # get the Layer 01 data
    mnist_lib.get_layer_01(ls_01, ld_01)
    l01_data = np.frombuffer(ffi.buffer(ld_01[0], ls_01.size * 4), dtype=np.float32)
    ls01_x = np.arange(0, ls_01.k, 1)

    l12_img = build_layer_image(ls_12, l12_data, [7, 19], 4, 1000)
    l08_img = build_layer_image(ls_08, l08_data, [6, 19], 2, 1000)

    result_div.text = """<p style="margin-top:8px; font-size:180px;">""" + "{}".format(res[0]) + """</p>"""

    source.data = {'input_img': [np.flipud(rgba_img.view("uint32").reshape(rgba_img.shape[:2]))], 'dnn_input': [np.flipud(dnn_img_view.view("uint32").reshape(dnn_img_view.shape[:2]))],
                   'l12_img': [np.flipud(l12_img)], 'l08_img': [np.flipud(l08_img)]}

    l02.renderers = []
    l02.vbar(x=ls02_x, bottom=0, top=l02_data, color='blue', width=0.2)
    l02.x_range.start = 0

    l01.renderers = []
    l01.vbar(x=ls01_x, bottom=0, top=l01_data, color='grey', width=0.5)
    l01.vbar(x=res[0], bottom=0, top=l01_data[res[0]], color='red', width=0.5)

    l01.xaxis.ticker = np.arange(0, 10)

def startup(doc):
    # the main entry point into the code
    # if __name__ == '__main__':
    jet_1k = jet_colormap(1000)

    init_mnist_lib()
    update()

    p1 = figure(plot_height=260, plot_width=250, title="Webcam Input", tools="save, pan, box_zoom, reset")
    p1.image_rgba(image="input_img", x=0, y=0, dw=400, dh=400, source=source)
    p1.axis.visible = False
    p1.grid.visible = False
    p1.x_range.range_padding = 0
    p1.y_range.range_padding = 0
    p1.title.text_font_size = "13pt"

    p2 = figure(plot_height=260, plot_width=250, title="DNN Input", tools="save, pan, box_zoom, reset")
    p2.image_rgba(image="dnn_input", x=0, y=0, dw=28, dh=28, source=source)
    p2.axis.visible = False
    p2.grid.visible = False
    p2.x_range.range_padding = 0
    p2.y_range.range_padding = 0
    p2.title.text_font_size = "13pt"

    l12 = figure(plot_height=500, plot_width=plot_width, title="Layer 12 Output")
    l12.image(image="l12_img", x=0, y=0, dw=400, dh=300, global_alpha=1.0, dilate=False, palette=jet_1k, source=source)
    l12.axis.visible = False
    l12.grid.visible = False
    l12.x_range.range_padding = 0
    l12.y_range.range_padding = 0
    l12.title.text_font_size = "13pt"

    l08 = figure(plot_height=500, plot_width=plot_width, title="Layer 08 Output")
    l08.image(image="l08_img", x=0, y=0, dw=400, dh=300, global_alpha=1.0, dilate=False, palette=jet_1k, source=source)
    l08.axis.visible = False
    l08.grid.visible = False
    l08.x_range.range_padding = 0
    l08.y_range.range_padding = 0
    l08.title.text_font_size = "13pt"

    l01.title.text_font_size = "13pt"
    l01.xaxis.major_label_text_font_size = "12pt"
    l01.xaxis.major_label_text_font_style = "bold"
    l01.yaxis.major_label_text_font_size = "12pt"
    l01.yaxis.major_label_text_font_style = "bold"

    l02.title.text_font_size = "13pt"
    l02.xaxis.major_label_text_font_size = "12pt"
    l02.xaxis.major_label_text_font_style = "bold"
    l02.yaxis.major_label_text_font_size = "12pt"
    l02.yaxis.major_label_text_font_style = "bold"

    #layout = column([row([column([p1, p2]), l12, l08]), row([Spacer(width=200, height=375), l02, l01])])

    first_row = row([Spacer(width=10, height=10), p1, Spacer(width=10, height=10), p2, arch_div, result_div])

    second_row = row([Spacer(width=10, height=10), l12, Spacer(width=10, height=10), l08])

    third_row = row([Spacer(width=10, height=10), l02, Spacer(width=10, height=10), l01])

    layout = column([first_row, second_row, third_row])

    #show(layout)

    # doc = curdoc()
    doc.title = "MNIST Viewer"
    doc.add_root(layout)
    doc.add_periodic_callback(update, update_time)

    # doc.hold('combine')

# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({'/': startup})
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
