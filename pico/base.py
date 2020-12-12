import numpy as np
from collections import OrderedDict


class Tensor(object):
    def __init__(self, data: np.ndarray, requires_grad=False) -> None:
        super(Tensor, self).__init__()
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        tracer.add_leaf(self)

    def __del__(self):
        del self.data

        pass

    def __str__(self) -> str:
        return '<pico.base.Tensor{}, requires_grad={}>'.format(self.data, self.requires_grad)

    def backward(self):
        tracer.backward(self, np.ones(1))

    def __add__(self, adder):
        add = Add()
        return add(self, adder)

    def __neg__(self):
        pass


class CTX(object):
    def __init__(self) -> None:
        super(CTX, self).__init__()
        self.saved_tensors = None

    def save_for_backward(self, *args):
        for arg in args:
            assert isinstance(
                arg, Tensor), "saving a non-Tensor type for backward is no support."
        self.saved_tensors = args

    def get_saved_tensors(self):
        return self.saved_tensors


class Function(object):
    def __init__(self) -> None:
        super(Function, self).__init__()

    @staticmethod
    def forward(ctx: CTX, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: CTX, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        id_func, ctx = tracer.add_func(self, *args, **kwargs)
        out = self.forward(ctx, *args, **kwargs)
        out.requires_grad = True
        tracer.add_tensor_from_func(out, id_func)
        return out


class Add(Function):
    def __init__(self) -> None:
        super(Add, self).__init__()

    @staticmethod
    def forward(ctx, tensorA: Tensor, tensorB: Tensor):

        return Tensor(tensorA.data + tensorB.data)

    @staticmethod
    def backward(ctx, grad_out: np.ndarray):
        return np.ones_like(grad_out), np.ones_like(grad_out)


class Tracer(object):
    def __init__(self) -> None:
        super(Tracer, self).__init__()
        self.tensors = OrderedDict()
        self.funcs = OrderedDict()

    def add_leaf(self, tensor: Tensor):
        for t, _ in self.tensors.values():
            assert t != tensor, "Trying to register a existing tensor to tracer!"

        if len(self.tensors) == 0:
            self.tensors[0] = (tensor, None)
        else:
            self.tensors[max(self.tensors.keys()) + 1] = (tensor, None)

    def add_tensor_from_func(self, tensor: Tensor, id_func):

        assert id_func in self.funcs
        self.tensors[self.index_tensor(tensor)] = (tensor, id_func)

    def rm_tensor(self, tensor: Tensor):
        idx = self.index_tensor(tensor)
        _, idx_func = self.tensors[idx]
        del self.tensor[idx]
        # if the function has no ref, del it
        save_flag = False
        for _, i in self.tensors.values():
            if i == idx_func:
                save_flag = True
        if not save_flag:
            self.rm_func(idx_func)
        pass

    def rm_func(self, id_func: int):
        if id_func is None:
            return
        _, ctx, _ = self.funcs[id_func]
        del ctx
        del self.funcs[id_func]
        # rm all refs
        for id_t, (t, id_f) in self.tensors.items():
            if id_f == id_func:
                self.tensors[id_t] = (t, None)

    def index_tensor(self, tensor: Tensor):
        for i, (t, _) in self.tensors.items():
            if t == tensor:
                return i

        raise ValueError("Unknow tensor to index.")

    def add_func(self, func: Function, *args, **kwargs):

        if len(self.funcs) == 0:
            idx = 0
        else:
            idx = max(self.funcs.keys()) + 1
        ctx = CTX()
        tensorlist = []
        for arg in args:
            if isinstance(arg, Tensor):
                tensorlist.append(self.index_tensor(arg))
            else:
                tensorlist.append(None)

        for arg in kwargs.values():
            if isinstance(arg, Tensor):
                tensorlist.append(self.index_tensor(arg))
            else:
                tensorlist.append(None)

        self.funcs[idx] = (func, ctx, tensorlist)
        return idx, ctx

    def backward(self, tensor: Tensor, grad: np.ndarray, idx_func=-1):
        if idx_func == -1:
            # unknown function id
            idx_tensor = self.index_tensor(tensor)
            _, idx_func = self.tensors[idx_tensor]
        else:
            assert idx_func in self.funcs or idx_func is None
            idx_func = idx_func

        if idx_func is None:
            return

        func, ctx, tensorList = self.funcs[idx_func]
        grad_outs = func.backward(ctx, grad)

        assert len(grad_outs) == len(
            tensorList), "{} not match the input arguments {}".format(grad_outs, tensorList)

        for g, t in zip(grad_outs, tensorList):
            if t is not None:
                t, idx_func_t = self.tensors[t]
                if t.requires_grad:
                    if t.grad is None:
                        t.grad = g
                    else:
                        t.grad += g
                self.backward(t, g, idx_func=idx_func_t)

        self.rm_func(idx_func)


tracer = Tracer()
