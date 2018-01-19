import numpy

from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class MaskReLU(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        self.mask = x[0] > 0
        return utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype)),

    def forward_gpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        self.mask = x[0] > 0
        y = cuda.cupy.maximum(x[0], 0)
        return y,

    def backward_cpu(self, x, gy):
        y = self.output_data[0]
        return utils.force_array(gy[0] * (y > 0)),

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = y > 0 ? gy : (T)0',
            'relu_bwd')(y, gy[0])
        return gx,


def mask_relu(x):
    y, = MaskReLU().apply((x,))
    return y
