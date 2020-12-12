from .base import Tensor, Function, CTX
import numpy as np


class MatMul(Function):
    def __init__(self) -> None:
        super(MatMul, self).__init__()

    @staticmethod
    def forward(ctx: CTX, A: Tensor, B: Tensor):
        ctx.save_for_backward(A, B)

        return Tensor(np.matmul(A.data, B.data))

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):
        A, B = ctx.get_saved_tensors()
        return np.matmul(grad_output, np.transpose(B.data)), np.matmul(np.transpose(A.data), grad_output)
