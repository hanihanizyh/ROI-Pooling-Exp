import mxnet as mx
from rcnn.config import config
from rcnn.symbol.params_view import ParamsView
from rcnn.symbol.stn_roi import ROIAffine
from rcnn.symbol.flat_roi import ExtractDim, FcEnsure
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1", attr={'force_mirroring': 'True'})
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2", attr={'force_mirroring': 'True'})
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1", attr={'force_mirroring': 'True'})
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2", attr={'force_mirroring': 'True'})
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1", attr={'force_mirroring': 'True'})
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2", attr={'force_mirroring': 'True'})
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3", attr={'force_mirroring': 'True'})
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1", attr={'force_mirroring': 'True'})
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2", attr={'force_mirroring': 'True'})
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3", attr={'force_mirroring': 'True'})
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1", attr={'force_mirroring': 'True'})
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2", attr={'force_mirroring': 'True'})
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3", attr={'force_mirroring': 'True'})

    return relu5_3


def get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name='data')
    im_info = mx.symbol.Variable(name='im_info')

    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name='rpn_conv_3x3')
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type='relu', name='rpn_relu', )
    rpn_cls_score = mx.symbol.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors,
                                          name='rpn_cls_score')
    rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors,
                                          name='rpn_bbox_pred')

    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name='rpn_cls_score_reshape')
    # rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True, normalization='valid', use_ignore=True, ignore_label=-1, name='rpn_cls_prob')

    # rpn_bbox_loss_ = rpn_bbox_weight*mx.symbol.smooth_l1(name='rpn_bbox_loss_', data=(rpn_bbox_pred-rpn_bbox_target), scalar=3.0)
    # rpn_bbox_loss = mx.symbol.Makeloss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale= 1.0/config.TRAIN.RPN_BATCH_SIZE)

    # ROI
    rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape, mode='channel', name='rpn_cls_act')
    rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0),
                                            name='rpn_cls_act_reshape')

    # rpn_cls_act_reshape = mx.symbol.Custom(data=rpn_cls_act_reshape, name='PView', op_type='PView')
    # rois = mx.symbol.contrib.Proposal(
    #     cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
    #     feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
    #     rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
    #     threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
    rois = mx.symbol.contrib.Proposal(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
        rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
        threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # rois = mx.symbol.Custom(data=rois, name='PView', op_type='PView')

    # gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    # group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target', \
    #                          num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES, \
    #                          batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)

    # rois = group[0]
    # # label = group[1]
    # # bbox_target = group[2]
    # # bbox_weight = group[3]
    #
    # rois = mx.symbol.Custom(data=rois, name='PView', op_type='PView')
    affine = mx.symbol.Custom(rpn_fm=relu5_3, rois=rois, op_type='RoiAffine', spatial_scale=config.RCNN_FEAT_STRIDE)
    roi_batch = mx.symbol.slice_axis(affine, axis=0, begin=0, end=1)
    roi_batch = mx.symbol.Custom(data=roi_batch, name='ExtractDim', op_type='ExtractDim')
    stn_pool = mx.symbol.SpatialTransformer(data=relu5_3, loc=roi_batch, target_shape=(7, 7), \
                                            transform_type='affine', sampler_type='bilinear')
    stn_pool = mx.symbol.Custom(data=stn_pool, name='PView', op_type='PView')
    # print '*'*10
    return stn_pool


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='stn_pool_test')
    parser.add_argument('--img', help='img_path', default='./testdata/000032.jpg', type=str)
    # parser.add_argument('--gt', help='gt_bbox', default=[], type=)
    args = parser.parse_args()
    return args


def get_image(img_path):
    img = cv.imread(img_path)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def get_gt():
    gt_boxes = np.array([[104, 78, 375, 183, 1], \
                         [133, 88, 197, 123, 1], \
                         [195, 180, 213, 229, 1], \
                         [26, 189, 44, 238, 1]])
    gt_boxes = gt_boxes[np.newaxis, :]
    return gt_boxes


def get_info():
    im_info = np.array([281, 500, 1])
    im_info = im_info[np.newaxis, :]
    return im_info


def plotgt(img_path, gt_boxes):
    im = cv.imread(img_path)
    plt.imshow(im)
    # box = gt_boxes[0][0]
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
# from ipdb import set_trace;
#
# set_trace()
# print mod.get_outputs()[0].asnumpy()

from collections import namedtuple
batch = namedtuple('batch', ['data'])
stn_pool = get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
# from ipdb import set_trace;set_trace()
mod = mx.mod.Module(symbol=stn_pool, data_names=('data','im_info'), label_names=None)
mod.bind(data_shapes=[('data', (1, 3, 281, 500)), ('im_info', (1, 3))])
mod.init_params()
# from ipdb import set_trace;set_trace()
mod.forward(batch(data=[mx.nd.array(img), mx.nd.array(im_info)]))
# from ipdb import set_trace;set_trace()
print mod.get_outputs()[0].asnumpy()

