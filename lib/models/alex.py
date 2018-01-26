import chainer
import chainer.links as L
import chainer.functions as F

from lib.functions import mask_relu
from lib.models import visualizer


class Alex(visualizer.Visualizer):

    def __init__(self):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

        visualizer._retrieve(
            'bvlc_alexnet.npz',
            'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
            self)

        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.fc6,
            self.fc7,
            self.fc8
        ]
        self.inv_layers = []
        self.mps = []
        for i in range(len(self.layers)):
            if i == 2 or i == 3 or i >= 5:
                self.mps.append(None)
            else:
                self.mps.append(F.MaxPooling2D(3, 2))
        self.relus = []
        for i in range(len(self.layers)):
            self.relus.append(mask_relu.MaskReLU())

    def feature_map_activations(self, x, layer_idx=None):
        if layer_idx is None:
            layer_idx = len(self.layers)

        pre_pooling_sizes = []
        h = x
        for i, (layer, relu, mp) in enumerate(
                zip(self.layers, self.relus, self.mps)):
            if i == layer_idx:
                break
            h = relu.apply((layer(h),))[0]
            if i == 0 or i == 1:
                h = F.local_response_normalization(h)
            if mp is not None:
                pre_pooling_sizes.append(h.data.shape[2:])
                # Disable cuDNN, else pooling indices will not be stored
                with chainer.using_config('use_cudnn', 'never'):
                    h = mp.apply((h,))[0]
            else:
                pre_pooling_sizes.append(None)

        return h, pre_pooling_sizes

    def activations(self, x, layer_idx, mask=True):
        self.check_add_inv_layers()
        h, unpooling_sizes = self.feature_map_activations(x, layer_idx)

        for i in reversed(range(layer_idx)):
            if self.mps[i] is not None:
                p = self.mps[i]
                h = F.upsampling_2d(
                    h, p.indexes, p.kh, p.sy, p.ph, unpooling_sizes[i])
            r = self.relus[i]
            if mask:
                relu_mask = r.mask
                h = self.inv_layers[i](r.apply((h,))[0] * relu_mask)
            else:
                h = self.inv_layers[i](r.apply((h,))[0])
            if i == 5:
                h = h.reshape(h.shape[0], 256, 6, 6)

        return h
