import numpy as np
from . import functional as F
from .base import Tensor
from collections import OrderedDict


class Parameter(Tensor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.retain = True


class Module(object):
    def __init__(self) -> None:
        super().__init__()
        self.training = True
        self._params_ = OrderedDict()
        self.sub_modules = OrderedDict()
        self.init_module()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self, attr='') -> str:
        return "{}({})".format(self.__class__.__name__, attr)

    def train(self):
        self.training = True
        for param in self.parameters():
            param.requires_grad = True

    def add_module(self, name, module):
        self.sub_modules[name] = module

    def eval(self):
        self.training = False
        for param in self.parameters():
            param.requires_grad = False

    def init_module(self):
        for name, attr in vars(self).items():
            if isinstance(attr, Parameter):
                self._params_[name] = attr
            elif isinstance(attr, Module):
                self.sub_modules[name] = attr

        for param in self.parameters():
            shapes = np.prod(param.size())
            param.data = np.random.randn(*(param.size())) * 0.1 / shapes

    def modules(self):
        yield self
        for module in self.sub_modules.values():
            yield module

    def named_modules(self):
        raise NotImplementedError

    def named_parameters(self, prefix=''):

        if prefix != '':
            if prefix[-1] != '.':
                prefix += '.'
        else:
            prefix = self.__class__.__name__ + '.'
        for name, param in self._params_.items():
            yield prefix + name, param

        for name, module in self.sub_modules.items():
            yield from module.named_parameters(prefix + name)

    def parameters(self):
        for module in self.modules():
            if module == self:
                yield from self._params_.values()
            else:
                yield from module.parameters()

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            [(name, param.state_dict())
             for name, param in self.named_parameters()]
        )

    def load_state_dict(self, ckpt: OrderedDict):
        for name, param in self.named_parameters():
            param.load_state_dict(ckpt[name])


class Sequential(Module):
    def __init__(self, ModuleList) -> None:
        super().__init__()
        for i, item in enumerate(ModuleList):
            if isinstance(item, Module):
                self.add_module(str(i), item)
            elif isinstance(item, tuple):
                name, module = item
                self.add_module(name, module)
            else:
                raise ValueError(
                    "Unknown type of item in ModuleList: {}".format(type(item)))

    def forward(self, *args, **kwargs):
        layers = list(self.sub_modules.values())
        output = layers[0](*args, **kwargs)
        # print(output.size())
        for l in layers[1:]:
            output = l(output)
            # print(l)
            # print(output.size())
        return output

    def __str__(self) -> str:
        attr = ['\n{}: {}'.format(name, module)
                for name, module in self.sub_modules.items()]
        attr = ''.join(attr)
        return super().__str__(attr=attr)


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):

        return F.ReLU(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weights = Parameter(
            np.empty((in_features, out_features)), requires_grad=True)
        self.bias = Parameter(
            np.empty(out_features), requires_grad=True
        )
        self.dims = (in_features, out_features)
        self.init_module()

    def forward(self, x):
        out = F.mm(x, self.weights) + self.bias
        return out

    def __str__(self) -> str:
        attr = 'in_features={}, out_features={}'.format(*(self.dims))
        return super().__str__(attr=attr)


class Conv2d(Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding=0) -> None:
        super().__init__()
        self.kernel = Parameter(np.empty(
            (kernel_size, kernel_size, in_features, out_features)), requires_grad=True)
        self.bias = Parameter(np.empty(out_features), requires_grad=True)

        self.dims = (in_features, out_features, kernel_size, stride, padding)

        self.init_module()

    def forward(self, x):
        _, _, _, stride, padding = self.dims
        out = F.Conv2D(x, self.kernel, stride, padding) + self.bias
        return out

    def __str__(self) -> str:
        return super().__str__(attr="in_features={}, out_features={}, kernel_size={}, stride={}, padding={}".format(*self.dims))


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        self.dims = (kernel_size, stride)

        self.init_module()

    def forward(self, x):
        out = F.MaxPool2d(x, self.dims[0], self.dims[1])
        return out

    def __str__(self) -> str:
        return super().__str__(attr="kernel_size={}, stride={}".format(*self.dims))


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        self.dims = (kernel_size, stride)

        self.init_module()

    def forward(self, x):
        out = F.AvgPool2d(x, self.dims[0], self.dims[1])
        return out

    def __str__(self) -> str:
        return super().__str__(attr="kernel_size={}, stride={}".format(*self.dims))


class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        out = F.flatten(x)
        return out
