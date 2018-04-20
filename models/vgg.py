import collections

import chainer
from chainer import functions as F

from lib import utils
from lib import traceable_nodes as T


class VGG(chainer.Chain):

    def __init__(self):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = T.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.conv1_2 = T.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2_1 = T.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv2_2 = T.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv3_1 = T.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv3_2 = T.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv3_3 = T.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv4_1 = T.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv4_2 = T.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv4_3 = T.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv5_1 = T.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv5_2 = T.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv5_3 = T.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.fc6 = T.Linear(None, 4096)
            self.fc7 = T.Linear(None, 4096)
            self.fc8 = T.Linear(None, 1000)

        utils._retrieve(
            'VGG_ILSVRC_16_layers.npz',
            'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/'
            'caffe/VGG_ILSVRC_16_layers.caffemodel',
            self)

        self.size = 224
        self.functions = collections.OrderedDict([
            ('conv1_1', [self.conv1_1, T.ReLU()]),
            ('conv1_2', [self.conv1_2, T.ReLU()]),
            ('pool1', [T.MaxPooling2D(2, 2)]),
            ('conv2_1', [self.conv2_1, T.ReLU()]),
            ('conv2_2', [self.conv2_2, T.ReLU()]),
            ('pool2', [T.MaxPooling2D(2, 2)]),
            ('conv3_1', [self.conv3_1, T.ReLU()]),
            ('conv3_2', [self.conv3_2, T.ReLU()]),
            ('conv3_3', [self.conv3_3, T.ReLU()]),
            ('pool3', [T.MaxPooling2D(2, 2)]),
            ('conv4_1', [self.conv4_1, T.ReLU()]),
            ('conv4_2', [self.conv4_2, T.ReLU()]),
            ('conv4_3', [self.conv4_3, T.ReLU()]),
            ('pool4', [T.MaxPooling2D(2, 2)]),
            ('conv5_1', [self.conv5_1, T.ReLU()]),
            ('conv5_2', [self.conv5_2, T.ReLU()]),
            ('conv5_3', [self.conv5_3, T.ReLU()]),
            ('pool5', [T.MaxPooling2D(2, 2)]),
            ('fc6', [self.fc6, T.ReLU(), F.dropout]),
            ('fc7', [self.fc7, T.ReLU(), F.dropout]),
            ('fc8', [self.fc8]),
            ('prob', [F.softmax]),
        ])

    def __call__(self, x, layer):
        h = x
        self.inv_functions = collections.OrderedDict()
        for key, funcs in self.functions.items():
            inv = []
            for func in funcs:
                inv.append(func)
                h = func(h)
            self.inv_functions[key] = inv
            if key == layer:
                break
        return h

    def activations(self, x, layer, mask=True):
        h = self(x, layer)
        for key, funcs in reversed(self.inv_functions.items()):
            for func in reversed(funcs):
                if isinstance(func, T.TraceableNode):
                    if isinstance(func, T.ReLU):
                        h = func.trace(h, mask)
                    else:
                        h = func.trace(h)
                else:
                    h = func(h)
        return h
