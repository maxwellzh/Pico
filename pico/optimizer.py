from .base import Tensor
import numpy as np
from collections import OrderedDict


class OPIM(object):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = {i: param for i, param in enumerate(list(params))}

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params.values():
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, ckpt: OrderedDict):
        for key, value in ckpt.items():
            setattr(self, key, value)


class SGD(OPIM):
    def __init__(self, params, lr: float, momentum: float = 0.) -> None:
        super().__init__(params)
        self.tracks_grads = OrderedDict(
            [(i, None) for i in self.params.keys()])
        self.momentum = momentum
        self.lr = lr

    def step(self):
        if self.momentum == 0.:
            for key, param in self.params.items():
                if param.grad is None or param.requires_grad is False:
                    continue
                param.data -= param.grad * self.lr
            return

        for key, param in self.params.items():
            if param.grad is None or param.requires_grad is False:
                continue

            if self.tracks_grads[key] is None:
                self.tracks_grads[key] = param.grad
            else:
                self.tracks_grads[key] = self.tracks_grads[key] * \
                    self.momentum + (1-self.momentum)*param.grad
            param.data -= self.tracks_grads[key] * self.lr

    def state_dict(self):

        return OrderedDict([('lr', self.lr), ('momentum', self.momentum), ('tracks_grads', self.tracks_grads)])


class Adam(OPIM):
    def __init__(self, params, lr: float, betas: tuple = (0.9, 0.999)) -> None:
        super().__init__(params)
        self.tracks_grad_m = OrderedDict(
            [(i, None) for i in self.params.keys()])
        self.tracks_grad_v = OrderedDict(
            [(i, None)for i in self.params.keys()])
        self.tracks_betas = OrderedDict(
            [(i, betas) for i in self.params.keys()])
        self.betas = betas
        self.lr = lr

    def step(self):
        for key, param in self.params.items():
            if param.grad is None or param.requires_grad is False:
                continue
            if self.tracks_grad_m[key] is None:
                self.tracks_grad_m[key] = param.grad
                self.tracks_grad_v[key] = param.grad ** 2
            else:
                self.tracks_grad_m[key] = self.tracks_grad_m[key] * \
                    self.betas[0] + (1-self.betas[0]) * param.grad
                self.tracks_grad_v[key] = self.tracks_grad_v[key] * \
                    self.betas[1] + (1-self.betas[1]) * param.grad ** 2

            m_hat = self.tracks_grad_m[key]/(1-self.tracks_betas[key][0])
            v_hat = self.tracks_grad_v[key]/(1-self.tracks_betas[key][1])

            param.data -= self.lr * m_hat / (1e-8 + np.sqrt(v_hat))

            self.tracks_betas[key] = (
                self.tracks_betas[key][0] * self.betas[0], self.tracks_betas[key][1] * self.betas[1])

    def state_dict(self):
        return OrderedDict([
            ('lr', self.lr),
            ('betas', self.betas),
            ('tracks_grad_m', self.tracks_grad_m),
            ('tracks_grad_v', self.tracks_grad_v),
            ('tracks_betas', self.tracks_betas)
        ])
