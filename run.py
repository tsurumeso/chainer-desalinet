import argparse

import chainer
import cv2
import numpy as np

import models


p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='images/bird.png')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--arch', '-a', choices=['alex', 'vgg'], default='alex')
p.add_argument('--mask', '-m', action='store_true')
args = p.parse_args()


if __name__ == '__main__':
    if args.arch == 'alex':
        model = models.Alex()
        layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                  'fc6', 'fc7', 'fc8']
    elif args.arch == 'vgg':
        model = models.VGG()
        layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3',
                  'fc6', 'fc7', 'fc8']

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    src = cv2.imread(args.input, 1)
    src = cv2.resize(src, (model.size, model.size))
    src = src.astype(np.float32) - np.float32([103.939, 116.779, 123.68])
    src = src.transpose(2, 0, 1)[np.newaxis, :, :, :]
    src = model.xp.array(src)

    for layer in layers:
        acts = model.trace(src, layer, mask=args.mask)
        dst = chainer.cuda.to_cpu(acts[0].data)
        dst = dst.transpose(1, 2, 0)
        dst -= dst.min()
        dst *= 255 / dst.max()
        cv2.imwrite(args.arch + '_' + layer + '.png', dst)
