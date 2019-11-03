import platform
import os
import time
from cffi import FFI
import numpy as np
import cv2 as cv
import bokeh
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, Spacer
from find_squares import find_squares

script_path = os.path.realpath(__file__)

# set up some global variables that will be used throughout the code
# read only
threshold = 150
dnn_alpha = np.full((28, 28), 255, dtype=np.uint8)
update_time = 50

ffi = FFI()

use_webcam = False
if use_webcam:
    vc = cv.VideoCapture(0)
else:
    image_name = os.path.dirname(os.path.dirname(script_path)) + "/input_test.png"

print("image path: " + str(image_name))

# modify these to point to the right locations
if platform.system() == "Windows":
    libname = "mnist_lib.dll"
    lib_location = "D:/Projects/mnist_dll/build/Release/" + libname
    weights_file = "D:/Projects/mnist_dll/nets/mnist_net_pso_14_97.dat"
elif platform.system() == "Linux":
    libname = "libmnist_lib.so"
    home = os.path.expanduser('~')
    lib_location = home + "/Projects/mnist_net_lib/build/" + libname
    weights_file = home + "/Projects/mnist_net_lib/nets/mnist_net_pso_14_97.dat"
else:
    quit()


mnist_lib = ffi.dlopen(lib_location)
print('Loaded lib {0}'.format(mnist_lib))

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
void get_layer_12(struct layer_struct *data, const float** data_params);
''')

def jet_clamp(v):
    v[v < 0] = 0
    v[v > 1] = 1
    return v

def jet_color(t, t_min, t_max):

    t_range = t_max - t_min
    t_avg = (t_max + t_min) / 2.0
    t_m = (t_max - t_avg) / 2.0

    rgb = np.empty((t.shape[0], t.shape[1], 4), dtype=np.uint8)
    rgb[:, :, 0] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg - t_m)))).astype(np.uint8)
    rgb[:, :, 1] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg)))).astype(np.uint8)
    rgb[:, :, 2] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg + t_m)))).astype(np.uint8)
    rgb[:, :, 3] = np.full((t.shape[0], t.shape[1]), 255, dtype=np.uint8)
    return rgb


mnist_lib.init_net(weights_file.encode('utf-8'))

color_img = cv.imread(image_name)
color_img = color_img[:, :, ::-1]

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

dnn_img = np.minimum(np.maximum((255 * (dnn_img - min_img)) / abs(max_img - min_img), 0), 255).astype(np.uint8)


# Allocate the pointers passed to run_net
res = ffi.new('unsigned int *')

mnist_lib.run_net(dnn_img.tobytes(), dnn_img.shape[0], dnn_img.shape[1], res)

print(res[0])

cm = color_img.tobytes()

# Allocate the pointers passed to get_layer
ls_12 = ffi.new('struct layer_struct*')
ld_12 = ffi.new('float**')
mnist_lib.get_layer_12(ls_12, ld_12)

l12_out = np.frombuffer(ffi.buffer(ld_12[0], ls_12.size*4), dtype=np.float32)
min_v = np.amin(l12_out)
max_v = np.amax(l12_out)
img_length = ls_12.nr * ls_12.nc

l12_1 = np.reshape(l12_out[0:img_length], [ls_12.nr, ls_12.nc])
l12_1 = cv.resize(l12_1, (20*ls_12.nr,20*ls_12.nc), interpolation=cv.INTER_NEAREST)


l12_all = np.reshape(l12_out, [ls_12.nr, ls_12.nc, ls_12.k], order='F')

t2 = []
for idx in range(ls_12.k):
    l12_1j = jet_color((l12_all[:,:,idx]).transpose(), min_v, max_v)
    l12_1j = cv.resize(l12_1j, (20*ls_12.nr,20*ls_12.nc), interpolation=cv.INTER_NEAREST)
    t2.append(cv.cvtColor(l12_1j, cv.COLOR_RGBA2BGRA))


# for idx in range(ls_12.k):
#     cv.imshow("test", t2[idx])
#     cv.waitKey(-1)

# cv.imshow("test2", l12_1)
# cv.waitKey(-1)

bp = 1


# create a random array and then turn into a jet image
t1 = time.perf_counter()
rnd_img = np.random.rand(10, 10)
rnd_img2 = [cv.resize(rnd_img, (200, 200), interpolation=cv.INTER_NEAREST)]
rnd_img_jet = jet_color(rnd_img2[0], 0, 1)
t2 = time.perf_counter()


elapsed_time = t2 - t1
print(elapsed_time)


# cv.imshow("test2", rnd_img_jet)
# cv.waitKey(-1)


bp = 2

source = ColumnDataSource(data=dict(input_img=[rnd_img_jet], rd=[rnd_img2[0]]))

p1 = figure(plot_height=500, plot_width=500, title="Input", toolbar_location="below", tools="pan,wheel_zoom,zoom_in,zoom_out,reset,box_select")
p1.image_rgba(image="input_img", x=0, y=0, dw=400, dh=400, source=source)
p1.axis.visible = False
p1.grid.visible = False
p1.x_range.range_padding = 0
p1.y_range.range_padding = 0


test = "@rd"

hover1 = [HoverTool(tooltips=[("value ", test)])]
p1.add_tools(hover1[0])


layout = row([p1])

show(layout)

