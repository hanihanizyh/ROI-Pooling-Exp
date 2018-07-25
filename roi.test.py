import mxnet as mx
from rcnn.config import config
from rcnn.symbol.params_view import ParamsView
from rcnn.symbol.stn_roi import ROIAffine
from rcnn.symbol.flat_roi import ExtractDim, FcEnsure
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name='data')
    # im_info = mx.symbol.Variable(name='im_info')
    gt_boxes = mx.symbol.Variable(name='gt_boxes')
    gt_boxes = mx.symbol.Custom(data=gt_boxes, name='ExtractDim', op_type='ExtractDim')
    # gt_boxes = mx.symbol.Custom(data=gt_boxes, name='PView', op_type='PView')
    # gt_boxes = mx.symbol.slice_axis(gt_boxes, axis=0, begin=0, end=1)

    # affine = mx.symbol.Custom(rpn_fm=data, rois=gt_boxes, op_type='RoiAffine', spatial_scale=1)
    # all_stn_pool = []
    # for n in range(int(4)):
    #     roi_batch = mx.symbol.slice_axis(affine, axis=0, begin=n, end=n+1)
    #     roi_batch = mx.symbol.Custom(data=roi_batch, name='ExtractDim', op_type='ExtractDim')
    #     stn_pool = mx.symbol.SpatialTransformer(data=data, loc=roi_batch, target_shape=(64, 64), \
    #                                             transform_type='affine', sampler_type='bilinear')
    #     # stn_pool = mx.symbol.Custom(data=stn_pool, name='PView', op_type='PView')
    #     all_stn_pool.append(stn_pool)
    # stn_pool_rois = mx.symbol.concat(*all_stn_pool, dim=0)
    # stn_pool_rois = mx.symbol.Custom(data=stn_pool_rois, name='PView', op_type='PView')
    # stn_pool_rois = mx.symbol.Custom(data=stn_pool_rois, name='PlotView', op_type='PlotView')
    # print '*'*10

    # stn_pool_rois = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=data, rois=gt_boxes, pooled_size=(64, 64), spatial_scale=1)
    stn_pool_rois = mx.contrib.symbol.ROIAlign(name='roi_pool5', data=data, rois=gt_boxes, pooled_size=(64, 64), spatial_scale=1)
    stn_pool_rois = mx.symbol.Custom(data=stn_pool_rois, name='PlotView', op_type='PlotView')

    return stn_pool_rois



import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='stn_pool_test')
    parser.add_argument('--img', help='img_path', default='./testdata/000032.jpg', type=str)
    args = parser.parse_args()
    return args


def get_image(img_path):
    img = cv.imread(img_path)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def get_gt():
    gt_boxes = np.array([[0, 104, 78, 375, 183], \
                         [0, 133, 88, 197, 123], \
                         [0, 195, 180, 213, 229], \
                         [0, 26, 189, 44, 238]])
    gt_boxes = gt_boxes[np.newaxis, :]
    return gt_boxes


def get_info():
    im_info = np.array([281, 500, 1])
    im_info = im_info[np.newaxis, :]
    return im_info


def plotgt(img_path, gt_boxes):
    im = cv.imread(img_path)
    plt.imshow(im)
    for box in gt_boxes[0]:
        plt.gca().add_patch(plt.Rectangle((box[1], box[2]), box[3] - box[1], box[4] - box[2], \
                                          fill=False, edgecolor='r', linewidth=3))
    plt.show()


args = parse_args()
gt_boxes = get_gt()
# plotgt(args.img,gt_boxes)

img = get_image(args.img)
gt_boxes = get_gt()
im_info = get_info()

from collections import namedtuple

batch = namedtuple('batch', ['data'])
stn_pool = get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
# from ipdb import set_trace;set_trace()
mod = mx.mod.Module(symbol=stn_pool, data_names=('data', 'gt_boxes'), label_names=None)
mod.bind(data_shapes=[('data', (1, 3, 281, 500)), ('gt_boxes', (1, 4, 5))])
mod.init_params()
# from ipdb import set_trace;set_trace()
mod.forward(batch(data=[mx.nd.array(img), mx.nd.array(gt_boxes)]))
# from ipdb import set_trace;set_trace()

# from collections import namedtuple
#
# batch = namedtuple('batch', ['data'])
# stn_pool = get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
# # from ipdb import set_trace;set_trace()
# mod = mx.mod.Module(symbol=stn_pool, data_names=('data', 'gt_boxes', 'im_info'), label_names=None)
# mod.bind(data_shapes=[('data', (1, 3, 281, 500)), ('gt_boxes', (1, 4, 5)), ('im_info', (1, 3))])
# mod.init_params()
# # from ipdb import set_trace;set_trace()
# mod.forward(batch(data=[mx.nd.array(img), mx.nd.array(gt_boxes), mx.nd.array(im_info)]))
# from ipdb import set_trace;set_trace()