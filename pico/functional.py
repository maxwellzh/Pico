from .base import Tensor, Function, CTX
import numpy as np


class _Dot_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, A: Tensor, B: Tensor):
        '''
        A: [..., n]
        B: [n]
        out: [..., 1]
        '''
        ctx.save_for_backward(A, B)

        return Tensor(np.dot(A.data, B.data))

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):
        '''
        grad_output: [..., 1]
        '''
        A, B = ctx.get_saved_tensors()

        # [..., n]
        grad_A = grad_output * B.data

        # [..., n]
        grad_B = grad_output * A.data

        # [n]
        grad_B = np.sum(grad_B, axis=tuple(range(A.dim()-1)))

        return grad_A, grad_B


class _MatMul_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, A: Tensor, B: Tensor):
        '''
        A: [..., n, m]
        B: [m, k]
        out: [..., n, k]
        '''
        ctx.save_for_backward(A, B)

        return Tensor(np.matmul(A.data, B.data))

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):
        '''
        grad_output: [..., n, k]
        '''
        A, B = ctx.get_saved_tensors()

        # [..., n, m]
        grad_A = np.matmul(grad_output, np.transpose(B.data))

        # [..., k, m]
        grad_B = np.matmul(np.transpose(
            A.data, (tuple(range(A.dim()-2))+(-1, -2))), grad_output)

        # [k, m]
        grad_B = np.sum(grad_B, axis=tuple(range(A.dim()-2)))

        return grad_A, grad_B


# class _ReLU_

class _EuclidLoss_(Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: CTX, item: Tensor, target: Tensor):
        ctx.res = item.data - target.data
        loss = np.linalg.norm(ctx.res)
        ctx.loss = loss
        return Tensor(loss)

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):
        _tmp = np.power(ctx.loss, -0.5) * ctx.res
        grad_item = _tmp * grad_output
        grad_target = -_tmp * grad_output
        return grad_item, grad_target


dot = _Dot_()
mm = _MatMul_()
EuclidLoss = _EuclidLoss_()
