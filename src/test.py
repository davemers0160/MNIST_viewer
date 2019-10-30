import platform
import os
import math
#import ctypes as ct
from cffi import FFI
import numpy as np
import cv2 as cv
import bokeh
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
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
void get_layer_01(struct layer_struct *data, const float** data_params);
''')


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
ls_01 = ffi.new('struct layer_struct*')
ld_01 = ffi.new('float**')
mnist_lib.get_layer_01(ls_01, ld_01)

arr_out = np.frombuffer(ffi.buffer(ld_01[0], ls_01.size*4), dtype=np.float32)


bp = 1
