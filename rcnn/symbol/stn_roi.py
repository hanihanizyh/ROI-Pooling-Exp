import mxnet as mx
from ipdb import set_trace

class ROIAffine(mx.operator.CustomOp):
    def __init__(self, spatial_scale):
        super(ROIAffine, self).__init__()
        self.spatial_scale = float(spatial_scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        rpn_fm = in_data[0]
        all_rois = in_data[1]
        # set_trace()
        batch_size = rpn_fm.shape[0]
        rois_size = all_rois.shape[0]
        ih = in_data[0].shape[2]
        iw = in_data[0].shape[3]
        # print ih
        # print iw
        # print('ih of img is %f, iw of img is %f' % (ih, iw))
        affine = mx.nd.zeros((rois_size, batch_size, 6), dtype='float16')
        for num, roi in enumerate(all_rois):
            roi = roi.asnumpy()
            roi[1] = max(roi[1] / self.spatial_scale, .0)
            roi[2] = max(roi[2] / self.spatial_scale, .0)
            roi[3] = min(max(roi[3], roi[1]+self.spatial_scale) / self.spatial_scale, iw-1)
            roi[4] = min(max(roi[4], roi[2]+self.spatial_scale) / self.spatial_scale, ih-1)
            x1 = -1 + 2 * roi[1] / (iw - 1)
            y1 = -1 + 2 * roi[2] / (ih - 1)
            x2 = -1 + 2 * roi[3] / (iw - 1)
            y2 = -1 + 2 * roi[4] / (ih - 1)
            aa = (x2 - x1) / 2
            bb = (x2 + x1) / 2
            cc = (y2 - y1) / 2
            dd = (y2 + y1) / 2
            # print ('aa = %f' % aa)
            # print ('bb = %f' % bb)
            # print ('cc = %f' % cc)
            # print ('dd = %f' % dd)
            affine[num][int(roi[0])] = [aa, .0, bb, .0, cc, dd]
        self.assign(out_data[0], req[0], affine)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('RoiAffine')
class ROIAffineProp(mx.operator.CustomOpProp):
    def __init__(self, spatial_scale):
        super(ROIAffineProp, self).__init__(need_top_grad=False)
        self.spatial_scale = spatial_scale

    def list_arguments(self):
        return ['rpn_fm', 'rois']

    def list_outputs(self):
        return ['affine-params']

    def infer_shape(self, in_shape):
        rpn_fm_shape = in_shape[0]
        rois_shape = in_shape[1]
        # print rpn_fm_shape, rois_shape
        affine_shape = (rois_shape[0], rpn_fm_shape[0], 6)

        return [rpn_fm_shape, rois_shape], [affine_shape]

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ROIAffine(self.spatial_scale)

# rpn_fm = mx.sym.Variable('rpn_fm')
# rois = mx.sym.Variable('rois')
# output = mx.sym.Custom(rpn_fm = rpn_fm, rois = rois, name='RoiAffine', op_type='RoiAffine')
# ex = output.simple_bind(mx.cpu(), rpn_fm=(2,3,4,4), rois=(2,5))
# y = ex.forward()
# set_trace()