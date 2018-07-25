import mxnet as mx
import numpy as np
from ipdb import set_trace
import cv2 as cv


class ParamsView(mx.operator.CustomOp):
    def __init__(self):
        super(ParamsView, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        params = in_data[0].asnumpy()
        print params.shape
        print params[0].shape
        # print params[1].shape
        # print params[2].shape
        # set_trace()
        # params = np.swapaxes(params, 0, 2)
        # params = np.swapaxes(params, 0, 1)
        # # print params.shape
        # params.astype(np.uint8)
        # cv.imwrite('./stn_later.jpg', params)
        # print params
        print '#'*10
        # set_trace()
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register('PView')
class ParamsViewProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ParamsViewProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, in_shape):
        # print 'pv', in_shape[0]
        return in_shape, in_shape
        # return in_shape, (in_shape[0],) * len(self.list_outputs()), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ParamsView()

# data = mx.sym.Variable('data')
# output = mx.sym.Custom(data=data, name='PView', op_type='PView')
# ex = output.simple_bind(mx.cpu(), data=(4,1))
# y = ex.forward()
# set_trace()
class CompareView(mx.operator.CustomOp):
    def __init__(self):
        super(CompareView, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        fm_data = in_data[0].asnumpy()
        stn_data = in_data[1].asnumpy()
        assert np.sum(abs(fm_data - stn_data)) == 0
        set_trace()
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register('CView')
class CompareViewProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CompareViewProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['fm_data','stn_data']

    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, in_shape):
        # print 'pv', in_shape[0]
        return in_shape[0], in_shape[0]
        # return in_shape, (in_shape[0],) * len(self.list_outputs()), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return CompareView()

class PlotView(mx.operator.CustomOp):
    def __init__(self):
        super(PlotView, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        params = in_data[0].asnumpy()
        print params.shape
        print params[0].shape
        # print params[1].shape
        # set_trace()
        # img = params[0]
        for i in range(int(4)):
            img = np.swapaxes(params[i], 0, 2)
            img = np.swapaxes(img, 0, 1)
            img.astype(np.uint8)
            cv.imwrite('/mnt/data-1/data/yuanhang.zhang/project/doing/fcnn.pointstn/rcnn/roi_pool_later%s.jpg' % str(i), img)
        # print params.shape
        # set_trace()
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register('PlotView')
class PlotViewProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(PlotViewProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, in_shape):
        # print 'pv', in_shape[0]
        return in_shape, in_shape
        # return in_shape, (in_shape[0],) * len(self.list_outputs()), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return PlotView()
