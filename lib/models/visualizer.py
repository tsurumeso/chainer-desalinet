import os

import chainer
from chainer import cuda
from chainer.dataset import download
from chainer.serializers import npz
import chainer.links as L
import chainer.functions as F


class Visualizer(chainer.Chain):

    def __init__(self):
        super(Visualizer, self).__init__()

    def __call__(self, x):
        h, _ = self.feature_map_activations(x)

        return F.softmax(h)

    def feature_map_activations(self, x, layer_idx=None):
        raise NotImplementedError()

    def check_add_inv_layers(self, nobias=True):
        if len(self.inv_layers) != 0:
            return

        for layer_block in self.layers:
            inv_layer_block = []
            for layer in layer_block:
                if isinstance(layer, L.Convolution2D):
                    out_channels, in_channels, kh, kw = layer.W.data.shape
                    if isinstance(layer.W.data, cuda.ndarray):
                        initialW = cuda.cupy.asnumpy(layer.W.data)
                    else:
                        initialW = layer.W.data
                    deconv = L.Deconvolution2D(
                        out_channels, in_channels, (kh, kw),
                        stride=layer.stride, pad=layer.pad, initialW=initialW,
                        nobias=nobias)
                    if isinstance(layer.W.data, cuda.ndarray):
                        deconv.to_gpu()
                    self.add_link('de{}'.format(layer.name), deconv)
                    inv_layer_block.append(deconv)
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
                    self.add_link('inv_{}'.format(layer.name), inv_fc)
                    inv_layer_block.append(inv_fc)
            self.inv_layers.append(inv_layer_block)


def _convert_caffemodel_to_npz(path_caffemodel, path_npz):
    # As CaffeFunction uses shortcut symbols,
    # we import CaffeFunction here.
    from chainer.links.caffe.caffe_function import CaffeFunction
    caffemodel = CaffeFunction(path_caffemodel)
    npz.save_npz(path_npz, caffemodel, compression=False)


def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    print('Now loading caffemodel (usually it may take few minutes)')
    _convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))