import collections

from chainer import functions as F

from lib import traceable_chain
from lib import traceable_nodes as T
from lib import utils


class Alex(traceable_chain.TraceableChain):

    def __init__(self):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = T.Convolution2D(3, 96, 11, stride=4)
            self.conv2 = T.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = T.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = T.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = T.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = T.Linear(None, 4096)
            self.fc7 = T.Linear(None, 4096)
            self.fc8 = T.Linear(None, 1000)

        utils._retrieve(
            'bvlc_alexnet.npz',
            'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
            self)

        self.size = 227
        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, T.ReLU(), T.LocalResponseNormalization()]),
            ('pool1', [T.MaxPooling2D(3, 2)]),
            ('conv2', [self.conv2, T.ReLU(), T.LocalResponseNormalization()]),
            ('pool2', [T.MaxPooling2D(3, 2)]),
            ('conv3', [self.conv3, T.ReLU()]),
            ('conv4', [self.conv4, T.ReLU()]),
            ('conv5', [self.conv5, T.ReLU()]),
            ('pool5', [T.MaxPooling2D(3, 2)]),
            ('fc6', [self.fc6, T.ReLU(), F.dropout]),
            ('fc7', [self.fc7, T.ReLU(), F.dropout]),
            ('fc8', [self.fc8]),
            ('prob', [F.softmax]),
        ])
