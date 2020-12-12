import numpy as np
from . import functional as F
from .base import Tensor
from collections import OrderedDict


class Module(object):
    def __init__(self) -> None:
        super().__init__()
        self.training = True
        self.params = OrderedDict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def init_weights(self):
        raise NotImplementedError

    def named_modules(self):
        raise NotImplementedError

    def named_parameters(self):
        return self.params.items()

    def parameters(self):
        return self.params.values()

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            [(name, param) for name, param in self.named_parameters()]
        )

    def load_state_dict(self, ckpt: OrderedDict):
        for name, param in self.named_parameters():
            setattr(self, name, ckpt[name])
            param.request_del()


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weights = Tensor(
            np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = Tensor(
            np.random.randn(out_features, 1), requires_grad=True
        )
        self.dims = (in_features, out_features)
        self.init_weights()

    def init_weights(self):
        self.weights.data = self.weights.data * \
            0.1 / self.dims[0] / self.dims[1]
        self.bias.data = self.bias.data * 0.1

        self.params = OrderedDict(
            [('weights', self.weights), ('bias', self.bias)])

    def forward(self, x):
        out = F.mm(x, self.weights) + self.bias
        return out


class Conv2d(Module):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__()
