import collections

import chainer

from lib import traceable_nodes as T


class TraceableChain(chainer.Chain):

    def __init__(self):
        super(TraceableChain, self).__init__()

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

    def trace(self, x, layer, mask=True):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
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
