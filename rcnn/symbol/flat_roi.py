import mxnet as mx

class ExtractDim(mx.operator.CustomOp):
    def __init__(self):
        super(ExtractDim, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        output = in_data[0][0]
        # from ipdb import set_trace;set_trace()
        self.assign(out_data[0], req[0], output)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register('ExtractDim')
class ExtractDimProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ExtractDimProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['extract_dim']

    def infer_shape(self, in_shape):
        extracted_shape = (in_shape[0][1], in_shape[0][2])
        return in_shape, [extracted_shape]

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ExtractDim()

class FcEnsure(mx.operator.CustomOp):
    def __init__(self):
        super(FcEnsure, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        output = mx.nd.zeros((128, 25088))
        self.assign(out_data[0], req[0], output)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register('FcEnsure')
class FcEnsureProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(FcEnsureProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [(128, 25088)]

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return FcEnsure()
