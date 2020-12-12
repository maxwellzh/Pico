import numpy as np
from collections import OrderedDict


class Tensor(object):
    def __init__(self, data: np.ndarray, requires_grad=False) -> None:
        super().__init__()
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        tracer.add_leaf(self)

    def state_dict(self):
        return OrderedDict([
            ('data', self.data),
            ('grad', self.grad),
            ('requires_grad', self.requires_grad),
        ])

    def load_state_dict(self, ckpt: OrderedDict):
        for key, value in ckpt.items():
            setattr(self, key, value)

    def __del__(self):
        del self.data

    def request_del(self):
        tracer.rm_tensor(self)

    def __str__(self) -> str:
        return 'Tensor({}, requires_grad={})'.format(self.data, self.requires_grad)

    def size(self):
        return self.data.shape

    def dim(self):
        return len(self.size())

    def backward(self, batchsize=1.):
        tracer.backward(self, np.ones_like(self.data)/batchsize)
        tracer.recycle(self)

    def clone(self):
        return Tensor(self.data, self.requires_grad)

    def __add__(self, adder):
        return base_oprator_add(self, adder)

    def __sub__(self, suber):
        return base_oprator_sub(self, suber)

    def __mul__(self, other):
        return base_oprator_mul(self, other)

    def __truediv__(self, other):
        return base_oprator_div(self, other)

    def __neg__(self):
        return base_oprator_neg(self)

    def __pos__(self):
        return base_oprator_pos(self)


class CTX(object):
    def __init__(self) -> None:
        super().__init__()
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
        super().__init__()

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


class _Add_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, tensorA: Tensor, tensorB: Tensor):
        ctx.size = tensorA.size(), tensorB.size()
        return Tensor(tensorA.data + tensorB.data)

    @staticmethod
    def backward(ctx: CTX, grad_out: np.ndarray):
        def _norm_axis_(grad: np.ndarray, target_size: tuple):
            if target_size != grad_out.shape:
                if sum(target_size) == len(target_size):
                    grad = np.sum(grad).reshape(target_size)
                else:
                    reduce_axis = tuple(
                        range(len(grad_out.shape)-len(target_size)))
                    grad = np.sum(grad, axis=reduce_axis)
            return grad

        grad_A, grad_B = grad_out, grad_out

        sizeA, sizeB = ctx.size

        grad_A = _norm_axis_(grad_A, sizeA)
        grad_B = _norm_axis_(grad_B, sizeB)

        return grad_A, grad_B


class _Sub_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, tensorA: Tensor, tensorB: Tensor):

        return Tensor(tensorA.data - tensorB.data)

    @staticmethod
    def backward(ctx: CTX, grad_out: np.ndarray):
        return grad_out, -grad_out


class _Mul_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, tensorA: Tensor, tensorB: Tensor):
        ctx.save_for_backward(tensorA, tensorB)
        return Tensor(tensorA.data * tensorB.data)

    @staticmethod
    def backward(ctx: CTX, grad_out: np.ndarray):
        A, B = ctx.get_saved_tensors()
        return B.data*grad_out, A.data*grad_out


class _Div_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, tensorA: Tensor, tensorB: Tensor):
        ctx.save_for_backward(tensorA, tensorB)
        return Tensor(tensorA.data / tensorB.data)

    @staticmethod
    def backward(ctx: CTX, grad_out: np.ndarray):
        A, B = ctx.get_saved_tensors()
        return 1./B.data * grad_out, (-A.data/(B.data ** 2))*grad_out


class _Neg_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, tensorA: Tensor):

        return Tensor(-tensorA.data)

    @staticmethod
    def backward(ctx: CTX, grad_out: np.ndarray):

        return -1.*grad_out


class _Pos_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, tensorA: Tensor):

        return Tensor(tensorA.data)

    @staticmethod
    def backward(ctx: CTX, grad_out: np.ndarray):

        return grad_out


class Tracer(object):
    def __init__(self) -> None:
        super().__init__()
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
        del self.tensors[idx]
        # if the function has no ref, del it
        save_flag = False
        for _, i in self.tensors.values():
            if i == idx_func:
                save_flag = True
        if not save_flag:
            self.rm_func(idx_func)

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

    def recycle(self, tensor: Tensor):
        while True:
            idx_tensor = self.index_tensor(tensor)
            _, idx_func = self.tensors[idx_tensor]
            if idx_func is None:
                break
            self.tensors[idx_tensor] = (tensor, None)

            refs = dict.fromkeys([id_f for _, id_f in self.tensors.values()])
            no_ref_funcs = [x for x in self.funcs.keys() if x not in refs]
            if len(no_ref_funcs) == 0:
                break
            idx_func = no_ref_funcs[0]
            _, _, tensorList = self.funcs[idx_func]
            self.rm_func(idx_func)
            tensorList = [x for x in tensorList if x is not None]
            func_refs = []
            for _, _, l in self.funcs.values():
                func_refs += l
            func_refs = dict.fromkeys(func_refs)

            for idx_tensor in tensorList:
                if idx_tensor not in func_refs:
                    tensor, _ = self.tensors[idx_tensor]
                    break

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
                    assert t.size() == g.shape, "tensor size {} not match the gradient size {}\ngenerate by {}".format(
                        t.size(), g.shape, func)
                    if t.grad is None:
                        t.grad = g
                    else:
                        t.grad += g
                    self.backward(t, g, idx_func=idx_func_t)


base_oprator_add = _Add_()
base_oprator_sub = _Sub_()
base_oprator_mul = _Mul_()
base_oprator_div = _Div_()
base_oprator_neg = _Neg_()
base_oprator_pos = _Pos_()

tracer = Tracer()
