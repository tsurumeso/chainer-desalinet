import chainer
import chainer.functions as F
from chainer.functions.normalization import local_response_normalization
from chainer.functions.pooling import max_pooling_2d
import chainer.links as L


from lib.functions import mask_relu


class TraceableNode(object):

    def trace(self):
        raise NotImplementedError()


class Convolution2D(L.Convolution2D, TraceableNode):

    def __init__(self, *args, **kwargs):
        super(Convolution2D, self).__init__(*args, **kwargs)
        self.inv = None
        self.pre_conv_size = None

    def __call__(self, x):
        self.pre_conv_size = x.shape[2:]
        return super(Convolution2D, self).__call__(x)

    def trace(self, x):
        if self.inv is None:
            out_channels, in_channels, kh, kw = self.W.data.shape
            initialW = chainer.backends.cuda.to_cpu(self.W.data)
            self.inv = L.Deconvolution2D(
                out_channels, in_channels, (kh, kw),
                stride=self.stride, pad=self.pad, initialW=initialW,
                nobias=True, outsize=self.pre_conv_size)
            if isinstance(self.W.data, chainer.backends.cuda.ndarray):
                self.inv.to_gpu()
        return self.inv(x)


class Linear(L.Linear, TraceableNode):

    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.inv = None
        self.pre_linear_size = None

    def __call__(self, x):
        self.pre_linear_size = x.shape
        return super(Linear, self).__call__(x)

    def trace(self, x):
        if self.inv is None:
            out_channels, in_channels = self.W.data.shape
            initialW = chainer.backends.cuda.to_cpu(self.W.data.T)
            self.inv = L.Linear(
                out_channels, in_channels, initialW=initialW, nobias=True)
            if isinstance(self.W.data, chainer.backends.cuda.ndarray):
                self.inv.to_gpu()
        return F.reshape(self.inv(x), self.pre_linear_size)


class MaxPooling2D(max_pooling_2d.MaxPooling2D, TraceableNode):

    def __init__(self, *args, **kwargs):
        super(MaxPooling2D, self).__init__(*args, **kwargs)
        self.pre_pooling_size = None

    def __call__(self, x):
        self.pre_pooling_size = x.shape[2:]
        with chainer.using_config('use_cudnn', 'never'):
            return self.apply((x,))[0]

    def trace(self, x):
        return F.upsampling_2d(
            x, self.indexes, self.kh, self.sy, self.ph, self.pre_pooling_size)


class ReLU(mask_relu.MaskReLU, TraceableNode):

    def __init__(self):
        super(ReLU, self).__init__()

    def __call__(self, x):
        return self.apply((x,))[0]

    def trace(self, x, mask):
        if mask:
            positive_mask = self.positive_mask
            return self.apply((x,))[0] * positive_mask
        else:
            return self.apply((x,))[0]


class LocalResponseNormalization(
        local_response_normalization.LocalResponseNormalization,
        TraceableNode):

    def __init__(self, *args, **kwargs):
        super(LocalResponseNormalization, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return super(LocalResponseNormalization, self).apply((x,))[0]

    def trace(self, x):
        return x
