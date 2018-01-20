import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F

from lib.functions import mask_relu


class Alex(chainer.Chain):

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

    def __call__(self, x):
        # Convolutional layers
        hs, _ = self.feature_map_activations(x)
        h = hs[-1]
        # # Fully connected layers
        # h = F.dropout(F.relu(self.fc6(h)))
        # h = F.dropout(F.relu(self.fc7(h)))
        # h = self.fc8(h)

        return F.softmax(h)

    def feature_map_activations(self, x, layer_idx=None):
        if layer_idx is None:
            layer_idx = len(self.layers)

        hs = []
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
            hs.append(h)

        return hs, pre_pooling_sizes

    def activations(self, x, layer_idx, mask=True):
        self.check_add_inv_layers()
        hs, unpooling_sizes = self.feature_map_activations(x, layer_idx)
        hs = [h.data for h in hs]
        h = hs[layer_idx - 1]

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

    def check_add_inv_layers(self, nobias=True):
        if len(self.inv_layers) != 0:
            return

        for layer in self.layers:
            if isinstance(layer, L.Convolution2D):
                out_channels, in_channels, kh, kw = layer.W.data.shape
                if isinstance(layer.W.data, cuda.ndarray):
                    initialW = cuda.cupy.asnumpy(layer.W.data)
                else:
                    initialW = layer.W.data
                deconv = L.Deconvolution2D(
                    out_channels, in_channels, (kh, kw), stride=layer.stride,
                    pad=layer.pad, initialW=initialW, nobias=nobias)
                if isinstance(layer.W.data, cuda.ndarray):
                    deconv.to_gpu()
                self.add_link('de{}'.format(layer.name), deconv)
                self.inv_layers.append(deconv)
            elif isinstance(layer, L.Linear):
                out_channels, in_channels = layer.W.data.shape
                if isinstance(layer.W.data, cuda.ndarray):
                    initialW = cuda.cupy.asnumpy(layer.W.data.T)
                else:
                    initialW = layer.W.data.T
                inv_fc = L.Linear(
                    out_channels, in_channels, initialW=initialW,
                    nobias=nobias)
                if isinstance(layer.W.data, cuda.ndarray):
                    inv_fc.to_gpu()
                self.add_link('de{}'.format(layer.name), inv_fc)
                self.inv_layers.append(inv_fc)
