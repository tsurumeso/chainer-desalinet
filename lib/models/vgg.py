import chainer
import chainer.links as L
import chainer.functions as F

from lib.functions import mask_relu
from lib.models import visualizer


class VGG(visualizer.Visualizer):

    def __init__(self):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv3_1 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv3_3 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv4_1 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv4_3 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv5_1 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv5_2 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv5_3 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

        visualizer._retrieve(
            'VGG_ILSVRC_16_layers.npz',
            'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/'
            'caffe/VGG_ILSVRC_16_layers.caffemodel',
            self)

        self.size = 224
        self.layers = [
            [self.conv1_1, self.conv1_2],
            [self.conv2_1, self.conv2_2],
            [self.conv3_1, self.conv3_2, self.conv3_3],
            [self.conv4_1, self.conv4_2, self.conv4_3],
            [self.conv5_1, self.conv5_2, self.conv5_3],
            [self.fc6],
            [self.fc7],
            [self.fc8]
        ]
        self.inv_layers = []
        self.mps = []
        for i in range(len(self.layers)):
            if i >= 5:
                self.mps.append(None)
            else:
                self.mps.append(F.MaxPooling2D(2, 2))
        self.relus = []
        for layer_block in self.layers:
            relu_block = []
            for _ in range(len(layer_block)):
                relu_block.append(mask_relu.MaskReLU())
            self.relus.append(relu_block)

    def feature_map_activations(self, x, layer_idx=None):
        if layer_idx is None:
            layer_idx = len(self.layers)

        pre_pooling_sizes = []
        h = x
        for i, (layer_block, relu_block, mp) in enumerate(
                zip(self.layers, self.relus, self.mps)):
            if i == layer_idx:
                break
            for layer, relu in zip(layer_block, relu_block):
                h = relu.apply((layer(h),))[0]
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
            for inv_layer, relu in zip(
                    reversed(self.inv_layers[i]), reversed(self.relus[i])):
                if mask:
                    relu_mask = relu.mask
                    h = inv_layer(relu.apply((h,))[0] * relu_mask)
                else:
                    h = inv_layer(relu.apply((h,))[0])
            if i == 5:
                h = h.reshape(h.shape[0], 512, 7, 7)

        return h