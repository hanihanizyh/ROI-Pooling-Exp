import mxnet as mx
import cv2 as cv
import numpy as np
from ipdb import set_trace
import matplotlib.pyplot as plot
from params_view import CompareView, ParamsView, ROIParams
from stn_roi import ROIAffine
from flat_roi import ExtractDim, FcEnsure
import time

import matplotlib.pyplot as plt

img_path = './liang.jpg'
def get_image(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (400, 600))
    cv.imwrite('./resized.jpg', img)
    img = cv.resize(img, (200, 300))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    # set_trace()
    return img

def get_roi():
    rois = np.array([0, 0, 0, 399, 299])
    rois = rois[np.newaxis, :]
    return rois

def get_roi_batch():
    rois = np.array([1, .0, .0, .0, 0.5, -0.5])
    rois = rois[np.newaxis, :]
    return rois

# set_trace()
img_data = mx.symbol.Variable('data')
roi_data = mx.symbol.Variable('rois')
roi_batch_data = mx.symbol.Variable('rois_batch')
#
img = get_image(img_path)
rois = get_roi()
# roi_batch = get_roi_batch()

affine = mx.symbol.Custom(rpn_fm=img_data, rois=roi_data, op_type='RoiAffine', spatial_scale=2)
roi_batch = mx.symbol.slice_axis(affine, axis=0, begin=0, end=1)
roi_batch = mx.symbol.Custom(data=roi_batch, name='ExtractDim', op_type='ExtractDim')
stn_pool = mx.symbol.SpatialTransformer(data=img_data, loc=roi_batch, target_shape=(150, 200), \
                                        transform_type="affine", sampler_type="bilinear")
stn_pool = mx.symbol.Custom(data=stn_pool, name='PView', op_type='PView')
# set_trace()
# whole test:
from collections import namedtuple
batch = namedtuple('batch', ['data'])
time0 = time.time()
mod = mx.mod.Module(symbol=stn_pool, data_names=('data', 'rois'), label_names=None)
time_mod = time.time() - time0
print (' time_mod : %f' % time_mod)
mod.bind(data_shapes=[('data', (1, 3, 300, 200)), ('rois', (1, 5))])
time_bind = time.time() - time_mod - time0
print (' time_bind : %f' % time_bind)
mod.init_params()
time_init = time.time() - time_bind - time_mod - time0
print (' time_init : %f' % time_init)
mod.forward(batch(data=[mx.nd.array(img), mx.nd.array(rois)]))
time_forward = time.time() - time_mod - time_bind - time_init - time0
print (' time_forward : %f' % time_forward)
# set_trace()

# stn test:
# from collections import namedtuple
# batch = namedtuple('batch', ['data'])
# mod = mx.mod.Module(symbol=stn_pool, data_names=('data', 'rois_batch'), label_names=None)
# mod.bind(data_shapes=[('data', (1, 3, 300, 200)), ('rois_batch', (1, 6))])
# mod.init_params()
# mod.forward(batch(data=[mx.nd.array(img), mx.nd.array(roi_batch)]))


# set_trace()
# print mod.get_outputs()[0].asnumpy()

# img1 = './liang.jpg'
# img2 = './stn_later.jpg'
#
# img1_data = cv.imread(img1)
# img2_data = cv.imread(img2)
#
# plt.subplot(211)
# plt.imshow(img1_data)
# plt.subplot(212)
# plt.imshow(img2_data)
# plt.show()


