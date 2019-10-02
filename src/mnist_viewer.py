import math
import ctypes as ct
import numpy as np
import cv2 as cv
import bokeh
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, Spacer


# set up some global variables that will be used throughout the code
# read only
threshold = 50
dnn_alpha = np.full((28, 28), 255, dtype=np.uint8)
image_name = "D:/Projects/mnist/data/test/wc_test.png"
vc = cv.VideoCapture(0)

# read and write global
mnist_dll = []
c1 = c2 = r1 = r2 = min_img = max_img = 0

source = ColumnDataSource(data=dict(input_img=[], dnn_input=[], l12_img=[], l08_img=[]))

l02 = figure(plot_height=350, plot_width=750, title="Layer 02")
l01 = figure(plot_height=350, plot_width=750, title="Layer 01")


class layer_struct(ct.Structure):
    _fields_ = [("k", ct.c_uint), ("n", ct.c_uint), ("nr", ct.c_uint), ("nc", ct.c_uint), ("size", ct.c_uint)]


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


def show_input(img):
    cm = bokeh.palettes.gray(256)

    p = figure(plot_height=150, plot_width=150)
    p.image(image=[np.flipud(np.array(img))], x=0, y=0, dw=img.width, dh=img.height, global_alpha=1.0, palette=cm)
    p.axis.visible = False
    p.grid.visible = False
    return p


def build_layer_image(ls, ld, cell_dim, padding, map_length, title):

    min_v = np.amin(ld)
    max_v = np.amax(ld)
    img_array = np.floor((map_length)*(ld - min_v)/(max_v - min_v)) + 100

    t_min = np.amin(img_array)
    t_max = np.amax(img_array)

    img_length = ls.nr * ls.nc

    img_h = (ls.nr + padding)*(cell_dim[0]-1) + ls.nr + 2*padding
    img_w = (ls.nc + padding)*(cell_dim[1]-1) + ls.nc + 2*padding
    # layer_img = np.zeros((img_h, img_w, 3), dtype=np.float)
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

    # p = figure(plot_height=300, plot_width=600, title=title)
    # p.image(image=[np.flipud(layer_img)], x=0, y=0, dw=img_w, dh=img_h, global_alpha=1.0, dilate=False, palette=jet_1k)
    # # p.image(image=[np.flipud(layer_img)], x=0, y=0, dw=img_w, dh=img_h, global_alpha=1.0, dilate=False, color_mapper=ColorMapper(palette=jet_1k, nan_color='black'))
    # p.axis.visible = False
    # p.grid.visible = False
    # p.x_range.range_padding = 0
    # p.y_range.range_padding = 0
    # return p
    return layer_img


def ld01_plot(x, y, title):
    res = np.argmax(y)
    p = figure(plot_height=200, plot_width=400, title=title)
    p.vbar(x=x, bottom=0, top=y, color='grey', width=0.5)
    p.vbar(x=res, bottom=0, top=y[res], color='red', width=0.5)
    p.xaxis.ticker = x
    return p


def ld_plot(x, y, title):
    p = figure(plot_height=200, plot_width=400, title=title)
    p.vbar(x=x, bottom=0, top=y, color='blue', width=0.2)
    return p


def init_mnist_dll():
    global mnist_dll, c1, c2, r1, r2, min_img, max_img
    mnist_dll = ct.cdll.LoadLibrary('D:/Projects/mnist_dll/build_dll/Release/MNIST_DLL.dll')

    # initialize the network with the weights file
    init_net = mnist_dll.init_net
    init_net.argtypes = [ct.c_char_p]
    init_net(ct.create_string_buffer(("D:/Projects/mnist_dll/nets/mnist_net_pso_14_97.dat").encode('utf-8')))

    # load in an image and convert to grayscale
    # color_img = cv.imread(image_name)
    ret, color_img = vc.read()
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    img_h = gray_img.shape[0]
    img_w = gray_img.shape[1]

    # find white box to bound the input
    c = np.argwhere(gray_img[math.floor(img_h / 2), :] > threshold)
    r = np.argwhere(gray_img[:, math.floor(img_w / 2)] > threshold)
    c1 = c.item(0) + 2
    c2 = c.item(len(c) - 1) - 2
    r1 = r.item(0) + 2
    r2 = r.item(len(r) - 1) - 2

    dnn_img = (255 - gray_img[r1:r2, c1:c2])
    dnn_img = cv.resize(dnn_img, (28, 28), interpolation=cv.INTER_CUBIC).astype('float')

    min_img = np.amin(dnn_img)
    max_img = np.amax(dnn_img)


def update():
    global mnist_dll, c1, c2, r1, r2, min_img, max_img, l01, l02, update_plot

    ret, color_img = vc.read()
    # color_img = cv.imread(image_name)
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    # crop out the white square
    dnn_img = (255 - gray_img[r1:r2, c1:c2])
    rgba_img = cv.cvtColor(color_img[r1:r2, c1:c2, :], cv.COLOR_BGR2RGBA)

    dnn_img = cv.resize(dnn_img, (28, 28), interpolation=cv.INTER_CUBIC).astype('float')

    dnn_img = (255 * (dnn_img - min_img) / (max_img - min_img)).astype(np.uint8)
    dnn_img_view = np.dstack([dnn_img, dnn_img, dnn_img, dnn_alpha])

    # img = Image.open(image_name).convert('L')
    # img2 = img.tobytes()

    # run the image on network and get the results
    # unsigned int run_net(unsigned char input[], unsigned int nr, unsigned int nc);
    run_net = mnist_dll.run_net
    run_net.argtypes = [ct.POINTER(ct.c_char), ct.c_uint32, ct.c_uint32]
    run_net.restype = ct.c_uint32
    res = run_net(dnn_img.tobytes(), dnn_img.shape[0], dnn_img.shape[1])

    # get the Layer 12 data
    # void get_layer_12(struct layer_struct *data, const float** data_params);
    get_layer_12 = mnist_dll.get_layer_12
    ls_12 = layer_struct()
    ld_12 = ct.POINTER(ct.c_float)()
    get_layer_12.argtypes = [ct.POINTER(layer_struct), ct.POINTER(ct.POINTER(ct.c_float))]
    get_layer_12(ct.byref(ls_12), ct.byref(ld_12))
    l12_data = np.ctypeslib.as_array(ld_12, [ls_12.size])

    # get the Layer 08 data
    # void get_layer_08(struct layer_struct *data, const float** data_params);
    get_layer_08 = mnist_dll.get_layer_08
    ls_08 = layer_struct()
    ld_08 = ct.POINTER(ct.c_float)()
    get_layer_08.argtypes = [ct.POINTER(layer_struct), ct.POINTER(ct.POINTER(ct.c_float))]
    get_layer_08(ct.byref(ls_08), ct.byref(ld_08))
    l08_data = np.ctypeslib.as_array(ld_08, [ls_08.size])

    # get the Layer 08 data
    # void get_layer_02(struct layer_struct *data, const float** data_params);
    get_layer_02 = mnist_dll.get_layer_02
    ls_02 = layer_struct()
    ld_02 = ct.POINTER(ct.c_float)()
    get_layer_02.argtypes = [ct.POINTER(layer_struct), ct.POINTER(ct.POINTER(ct.c_float))]
    get_layer_02(ct.byref(ls_02), ct.byref(ld_02))
    l02_data = np.ctypeslib.as_array(ld_02, [ls_02.size])
    ls02_x = np.arange(0, ls_02.k, 1)

    # get the Layer 01 data
    # void get_layer_01(struct layer_struct *data, const float** data_params);
    get_layer_01 = mnist_dll.get_layer_01
    ls_01 = layer_struct()
    ld_01 = ct.POINTER(ct.c_float)()
    get_layer_01.argtypes = [ct.POINTER(layer_struct), ct.POINTER(ct.POINTER(ct.c_float))]
    get_layer_01(ct.byref(ls_01), ct.byref(ld_01))
    l01_data = np.ctypeslib.as_array(ld_01, [ls_01.k])
    ls01_x = np.arange(0, ls_01.k, 1)
    l01_res = np.zeros(ls_01.k)
    l01_res[res] = l01_data.item(res)

    l12_img = build_layer_image(ls_12, l12_data, [7, 19], 4, 1000, "Layer 12")
    l08_img = build_layer_image(ls_08, l08_data, [6, 19], 2, 1000, "Layer 08")

    # l01_d = {'ls': ls01_x, 'a': l01_data, 'b': l01_res}

    # source.data = {'input_img': [np.flipud(rgba_img)], 'dnn_input': [np.flipud(dnn_img_view)],
    #                'l12_img': [np.flipud(l12_img)], 'l08_img': [np.flipud(l08_img)],
    #                'l02_x': [ls02_x], 'l02_y': [l02_data],
    #                'l01_x': [ls01_x], 'l01_y': [l01_data], 'l01_res': [res], 'l01_res_y': [l01_data[res]]}

    source.data = {'input_img': [np.flipud(rgba_img)], 'dnn_input': [np.flipud(dnn_img_view)],
                   'l12_img': [np.flipud(l12_img)], 'l08_img': [np.flipud(l08_img)]}

    l02.renderers = []
    l02.vbar(x=ls02_x, bottom=0, top=l02_data, color='blue', width=0.2)
    l02.x_range.start = 0

    l01.renderers = []
    l01.vbar(x=ls01_x, bottom=0, top=l01_data, color='grey', width=0.5)
    l01.vbar(x=res, bottom=0, top=l01_data[res], color='red', width=0.5)

    l01.xaxis.ticker = np.arange(0, 10)


# the main entry point into the code
# if __name__ == '__main__':
jet_1k = jet_colormap(1000)

init_mnist_dll()
update()

p1 = figure(plot_height=250, plot_width=200, title="Webcam Input", toolbar_location="below")
p1.image_rgba(image="input_img", x=0, y=0, dw=400, dh=400, source=source)
p1.axis.visible = False
p1.grid.visible = False
p1.x_range.range_padding = 0
p1.y_range.range_padding = 0

p2 = figure(plot_height=250, plot_width=200, title="DNN Input", toolbar_location="below")
p2.image_rgba(image="dnn_input", x=0, y=0, dw=28, dh=28, source=source)
p2.axis.visible = False
p2.grid.visible = False
p2.x_range.range_padding = 0
p2.y_range.range_padding = 0

l12 = figure(plot_height=500, plot_width=750, title="Layer 12")
l12.image(image="l12_img", x=0, y=0, dw=400, dh=300, global_alpha=1.0, dilate=False, palette=jet_1k, source=source)
l12.axis.visible = False
l12.grid.visible = False
l12.x_range.range_padding = 0
l12.y_range.range_padding = 0

l08 = figure(plot_height=500, plot_width=750, title="Layer 08")
l08.image(image="l08_img", x=0, y=0, dw=400, dh=300, global_alpha=1.0, dilate=False, palette=jet_1k, source=source)
l08.axis.visible = False
l08.grid.visible = False
l08.x_range.range_padding = 0
l08.y_range.range_padding = 0

layout = column([row([column([p1, p2]), l12, l08]), row([Spacer(width=200, height=375), l02, l01])])

show(layout)

doc = curdoc()
doc.title = "MNIST Viewer"
doc.add_root(layout)
doc.add_periodic_callback(update, 400)