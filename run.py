import argparse

import chainer
import cv2
import numpy as np

from lib import models


p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='images/bird.png')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--mask', '-m', action='store_true')
args = p.parse_args()


if __name__ == '__main__':
    model = models.Alex()
    chainer.serializers.load_npz('alex.npy', model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    src = cv2.imread(args.input, 1)
    src = cv2.resize(src, (227, 227))
    h, w = src.shape[:2]
    src = src.transpose(2, 0, 1)[np.newaxis, :, :, :]
    src = src.astype(np.float32)
    src = model.xp.array(src)

    for idx in range(1, 9):
        acts = model.activations(src, idx, mask=args.mask)
        dst = chainer.cuda.to_cpu(acts[0].data)
        dst = dst.transpose(1, 2, 0)
        dst -= dst.min()
        dst *= 255 / dst.max()
        dst = cv2.resize(dst, (w, h))
        if idx <= 5:
            cv2.imwrite('conv{}.png'.format(idx), dst)
        else:
            cv2.imwrite('fc{}.png'.format(idx), dst)
