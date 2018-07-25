# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from rcnn.config import config
from . import proposal
from . import proposal_target
from params_view import ParamsView
from stn_roi import ROIAffine
from flat_roi import ExtractDim, FcEnsure
from ipdb import set_trace

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

def get_vgg_rcnn(num_classes=config.NUM_CLASSES):
    """
    Fast R-CNN with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # reshape input
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
    label = mx.symbol.Reshape(data=label, shape=(-1, ), name='label_reshape')
    bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_classes), name='bbox_target_reshape')
    bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_classes), name='bbox_weight_reshape')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss])
    return group


def get_vgg_rcnn_test(num_classes=config.NUM_CLASSES):
    """
    Fast R-CNN Network with VGG
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')

    # reshape rois
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

    # shared convolutional layer
    relu5_3 = get_vgg_conv(data)

    # # Fast R-CNN

    affine = mx.symbol.Custom(rpn_fm=relu5_3, rois=rois, op_type='RoiAffine', spatial_scale=config.RCNN_FEAT_STRIDE)
    # affine = mx.symbol.Custom(data=affine, name='PView', op_type='PView')
    # pool5 = affine
    all_stn_pool = []
    # for n in range(int(config.TRAIN.BATCH_ROIS / 2)):
    for n in range(int(config.TRAIN.BATCH_ROIS)):
        roi_batch = mx.symbol.slice_axis(affine, axis=0, begin=n, end=n + 1)
        # roi_batch = mx.symbol.Custom(data=roi_batch, name='PView', op_type='PView')
        roi_batch = mx.symbol.Custom(data=roi_batch, name='ExtractDim', op_type='ExtractDim')
        # roi_batch = mx.symbol.Custom(data=roi_batch, name='PView', op_type='PView')
        stn_pool = mx.symbol.SpatialTransformer(data=relu5_3, loc=roi_batch, target_shape=(7, 7), \
                                                transform_type="affine", sampler_type="bilinear")
        # stn_pool = mx.symbol.Custom(data=stn_pool, name='PView', op_type='PView')
        all_stn_pool.append(stn_pool)
    stn_pool_rois = mx.symbol.concat(*all_stn_pool, dim=0)
    # stn_pool_rois = mx.symbol.Custom(data=stn_pool_rois, name='PView', op_type='PView')
    pool5 = stn_pool_rois
    # pool5 = mx.symbol.Custom(data=pool5, name='PView', op_type='PView')

    # pool5 = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # pool5 = mx.symbol.ROIAlign(name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([cls_prob, bbox_pred])
    return group


def get_vgg_rpn(num_anchors=config.NUM_ANCHORS):
    """
    Region Proposal Network with VGG
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1, name="cls_prob")
    # bounding box regression
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss])
    return group


def get_vgg_rpn_test(num_anchors=config.NUM_ANCHORS):
    """
    Region Proposal Network with VGG
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        group = mx.symbol.contrib.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE)
    else:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE)
    # rois = group[0]
    # score = group[1]

    return group


def get_vgg_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.symbol.contrib.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN

    affine = mx.symbol.Custom(rpn_fm=relu5_3, rois=rois, op_type='RoiAffine', spatial_scale=config.RCNN_FEAT_STRIDE)
    # affine = mx.symbol.Custom(data=affine, name='PView', op_type='PView')
    # pool5 = affine
    all_stn_pool = []
    for n in range(int(config.TRAIN.BATCH_ROIS)):
        roi_batch = mx.symbol.slice_axis(affine, axis=0, begin=n, end=n + 1)
        # roi_batch = mx.symbol.Custom(data=roi_batch, name='PView', op_type='PView')
        roi_batch = mx.symbol.Custom(data=roi_batch, name='ExtractDim', op_type='ExtractDim')
        # roi_batch = mx.symbol.Custom(data=roi_batch, name='PView', op_type='PView')
        stn_pool = mx.symbol.SpatialTransformer(data=relu5_3, loc=roi_batch, target_shape=(7, 7), \
                                                transform_type="affine", sampler_type="bilinear")
        # stn_pool = mx.symbol.Custom(data=stn_pool, name='PView', op_type='PView')
        all_stn_pool.append(stn_pool)
    stn_pool_rois = mx.symbol.concat(*all_stn_pool, dim=0)
    stn_pool_rois = mx.symbol.Custom(data=stn_pool_rois, name='PView', op_type='PView')
    pool5 = stn_pool_rois

    ### ROI Pooling
    # pool5 = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # #pool5 = mx.symbol.Custom(data=pool5, name='PView', op_type='PView')

    ### ROI Align
    # pool5 = mx.symbol.ROIAlign(name='roi_pool5', data=relu5_3, rois=rois, sample_per_part=2, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    #

    ### DeformablePSROIPooling
    # offset_t = mx.contrib.symbol.DeformablePSROIPooling(name='offset_t', data=relu5_3, rois=rois, group_size=1,
    #                                                  pooled_size=7,
    #                                                  sample_per_part=4, no_trans=True, part_size=7, output_dim=512,
    #                                                  spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # offset = mx.symbol.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
    # offset_reshape = mx.symbol.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")
    #
    # pool5 = mx.contrib.symbol.DeformablePSROIPooling(name='deformable_roi_pool', data=relu5_3,
    #                                                             rois=rois,
    #                                                             trans=offset_reshape, group_size=1, pooled_size=7,
    #                                                             sample_per_part=4,
    #                                                             no_trans=False, part_size=7, output_dim=512,
    #                                                             spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, trans_std=0.1)
    #

    ### PSROIPooling
    # pool5 = mx.contrib.symbol.PSROIPooling(name='ps_roi_pool', data=relu5_3,
    #                                                             rois=rois,
    #                                                             group_size=1, pooled_size=7,
    #                                                             output_dim=512,
    #

    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group


def get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)
    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu", attr={'force_mirroring': 'True'})
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.contrib.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    # rois = mx.symbol.Custom(data=rois, name='PView', op_type='PView')
    # gt_boxes = mx.symbol.Custom(data=gt_boxes, name='PView', op_type='PView')
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]


    # Fast R-CNN

    rois = mx.symbol.Custom(data=rois, name='PView', op_type='PView')
    # relu5_3 = mx.symbol.Custom(data=relu5_3, name='PView', op_type='PView')

    ### STN Pooling
    affine = mx.symbol.Custom(rpn_fm=relu5_3, rois=rois, op_type='RoiAffine', spatial_scale=config.RCNN_FEAT_STRIDE)
    # affine = mx.symbol.Custom(data=affine, name='PView', op_type='PView')
    # pool5 = affine
    all_stn_pool = []
    for n in range(int(config.TRAIN.BATCH_ROIS)):
        roi_batch = mx.symbol.slice_axis(affine, axis=0, begin=n, end=n+1)
        #roi_batch = mx.symbol.Custom(data=roi_batch, name='PView', op_type='PView')
        roi_batch = mx.symbol.Custom(data=roi_batch, name='ExtractDim', op_type='ExtractDim')
        #roi_batch = mx.symbol.Custom(data=roi_batch, name='PView', op_type='PView')
        stn_pool = mx.symbol.SpatialTransformer(data=relu5_3, loc=roi_batch, target_shape=(7, 7),\
                                  transform_type="affine", sampler_type="bilinear")
        # stn_pool = mx.symbol.Custom(data=stn_pool, name='PView', op_type='PView')
        all_stn_pool.append(stn_pool)
    stn_pool_rois = mx.symbol.concat(*all_stn_pool, dim=0)
    stn_pool_rois = mx.symbol.Custom(data=stn_pool_rois, name='PView', op_type='PView')
    pool5 = stn_pool_rois
    # pool5 = mx.symbol.Custom(data=pool5, name='PView', op_type='PView')

    ### ROI Pooling
    # pool5 = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # #pool5 = mx.symbol.Custom(data=pool5, name='PView', op_type='PView')

    ### ROI Align
    # pool5 = mx.symbol.ROIAlign(name='roi_pool5', data=relu5_3, rois=rois, sample_per_part=2, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    #

    ### DeformablePSROIPooling
    # offset_t = mx.contrib.symbol.DeformablePSROIPooling(name='offset_t', data=relu5_3, rois=rois, group_size=1,
    #                                                  pooled_size=7,
    #                                                  sample_per_part=4, no_trans=True, part_size=7, output_dim=512,
    #                                                  spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # offset = mx.symbol.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
    # offset_reshape = mx.symbol.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")
    #
    # pool5 = mx.contrib.symbol.DeformablePSROIPooling(name='deformable_roi_pool', data=relu5_3,
    #                                                             rois=rois,
    #                                                             trans=offset_reshape, group_size=1, pooled_size=7,
    #                                                             sample_per_part=4,
    #                                                             no_trans=False, part_size=7, output_dim=512,
    #                                                             spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, trans_std=0.1)
    #

    ### PSROIPooling
    # pool5 = mx.contrib.symbol.PSROIPooling(name='ps_roi_pool', data=relu5_3,
    #                                                             rois=rois,
    #                                                             group_size=1, pooled_size=7,
    #                                                             output_dim=512,
    #                                                             spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")

    # flatten = mx.symbol.Custom(data=pool5, name='FcEnsure', op_type='FcEnsure')

    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")


    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6", attr={'force_mirroring': 'True'})
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7", attr={'force_mirroring': 'True'})
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group
