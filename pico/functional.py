from .base import Tensor, Function, CTX

import numpy as np


class _Dot_(Function):
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


class _ReLU_(Function):
    @staticmethod
    def forward(ctx: CTX, tensor: Tensor):
        ctx.mark_pos = tensor.data > 0.
        return Tensor(tensor.data * ctx.mark_pos)

    @staticmethod
    def backward(ctx: CTX, grad_out: np.ndarray):
        pos = ctx.mark_pos
        return pos * grad_out


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


def _stride_select_(x: np.ndarray, kernel_size: int, stride: int):
    r'''
    input: `x`, [N, H, W, C]
    return `out`
    `out` [N, OH, OW, kernel_size, kernel_size, C]
    '''
    N, H, W, C = x.shape
    h1 = kernel_size
    w1 = kernel_size
    OH, OW = 0, 0
    while w1 <= W:
        OW += 1
        w1 += stride
    while h1 <= H:
        OH += 1
        h1 += stride
    s_n, s_h, s_w, s_c = x.strides
    new_strides = (s_n, s_h*stride, s_w*stride, s_h, s_w, s_c)
    reshape_x = np.lib.stride_tricks.as_strided(
        x, (N, OH, OW, kernel_size, kernel_size, C), new_strides)
    return reshape_x


class _Conv2D_(Function):

    # layout [N, H, W, C] x [H_k, W_k, C, K]
    @staticmethod
    def forward(ctx: CTX, feature_map: Tensor, kernel: Tensor, stride, padding):

        N, H, W, C = feature_map.data.shape
        H_k, W_k, _, K = kernel.data.shape
        assert H_k == W_k

        im = np.pad(feature_map.data, ((0, 0), (padding, padding),
                                       (padding, padding), (0, 0)), 'constant')
        reshaped_im = _stride_select_(im, H_k, stride)

        N, OH, OW, h_k, w_k, C = reshaped_im.shape

        reshaped_im = reshaped_im.reshape(N, OH, OW, h_k * w_k * C)
        kernel_data = kernel.data.reshape(H_k * W_k * C, K)

        # [N, OH, OW, K]
        out = Tensor(np.matmul(reshaped_im, kernel_data))
        ctx.attrs = [im.shape, kernel.size(), stride, padding]
        ctx.kernel = kernel_data
        ctx.reshaped_x = reshaped_im
        return out

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):
        # grad_output: [N, OH, OW, K]

        # kernel: [H_k * W_k * C, K]
        kernel = ctx.kernel
        reduced_dim = kernel.shape[0]

        # reshape_x: [N, OH, OW, h_k * w_k * C]
        reshaped_x = ctx.reshaped_x

        padded_size, kernel_size, stride, padding = ctx.attrs
        grad_x = np.zeros(padded_size)
        map_x = _stride_select_(grad_x, kernel_size[0], stride)
        map_x += (grad_output @ np.transpose(kernel)).reshape(map_x.shape)
        if padding != 0:
            grad_x = grad_x[:, padding:-padding, padding:-padding, :]

        grad_kernel = np.transpose(
            reshaped_x.reshape(-1, reduced_dim)) @ grad_output.reshape(-1, grad_output.shape[-1])
        grad_kernel = grad_kernel.reshape(kernel_size)

        return grad_x, grad_kernel, None, None


class _MaxPool2D_(Function):
    @staticmethod
    def forward(ctx: CTX, feature_map: Tensor, kernel_size, stride):

        # [N, OH, OW, h_k, w_k, C]
        reshaped_im = _stride_select_(feature_map.data, kernel_size, stride)
        N, OH, OW, h_k, w_k, C = reshaped_im.shape
        # [N, OH, OW, C]
        out = np.max(reshaped_im, axis=(3, 4))

        mask = np.repeat(out, h_k*w_k, axis=3).reshape(N, OH, OW, h_k, w_k, C)
        mask = mask >= reshaped_im
        ctx.mask = mask
        ctx.dims = (N, OH, OW, h_k, w_k, C)
        ctx.attrs = (feature_map.size(), kernel_size, stride)
        return Tensor(out)

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):
        mask = ctx.mask
        N, OH, OW, h_k, w_k, C = ctx.dims
        grad_in = np.repeat(grad_output, h_k*w_k,
                            axis=3).reshape(N, OH, OW, h_k, w_k, C)
        mask_grad = grad_in * mask

        feature_size, kernel_size, stride = ctx.attrs
        grad_in = np.zeros(feature_size)
        map_grad = _stride_select_(grad_in, kernel_size, stride)
        map_grad += mask_grad

        return grad_in, None, None


class _AvgPool2D_(Function):
    @staticmethod
    def forward(ctx: CTX, feature_map: Tensor, kernel_size, stride):

        # [N, OH, OW, h_k, w_k, C]
        reshaped_im = _stride_select_(feature_map.data, kernel_size, stride)
        N, OH, OW, h_k, w_k, C = reshaped_im.shape
        # [N, OH, OW, C]
        out = np.mean(reshaped_im, axis=(3, 4))

        ctx.dims = (N, OH, OW, h_k, w_k, C)
        ctx.attrs = (feature_map.size(), kernel_size, stride)
        return Tensor(out)

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):
        N, OH, OW, h_k, w_k, C = ctx.dims
        grad_in = np.repeat(grad_output, h_k*w_k,
                            axis=3).reshape(N, OH, OW, h_k, w_k, C)

        mask_grad = grad_in / h_k / w_k

        feature_size, kernel_size, stride = ctx.attrs
        grad_in = np.zeros(feature_size)
        map_grad = _stride_select_(grad_in, kernel_size, stride)
        map_grad += mask_grad

        return grad_in, None, None


class _Flatten_(Function):
    @staticmethod
    def forward(ctx: CTX, tensor: Tensor):
        # [N, ...]
        ctx.size = tensor.size()
        out = tensor.data.reshape(tensor.size()[0], -1)
        return Tensor(out)

    @staticmethod
    def backward(ctx: CTX, grad_output: np.ndarray):

        return grad_output.reshape(ctx.size)


class _CrossEntropyLoss_(Function):
    @staticmethod
    def forward(ctx: CTX, input: Tensor, target: Tensor):
        assert input.size()[0] == target.size()[0]

        exp_x = np.exp(input.data)
        loss = -np.log(exp_x[list(range(input.size()[0])),
                             target.data]/np.sum(exp_x, axis=1))
        loss = np.mean(loss)
        ctx.label = target.data
        ctx.exp_x = exp_x
        return Tensor(loss)

    @staticmethod
    def backward(ctx: CTX, grad_output):
        exp_x = ctx.exp_x
        label = ctx.label
        one_hot = np.eye(exp_x.shape[0], exp_x.shape[1])[label]

        return grad_output * (exp_x - one_hot) / exp_x.shape[0], None


dot = _Dot_()
mm = _MatMul_()
EuclidLoss = _EuclidLoss_()
ReLU = _ReLU_()
Conv2D = _Conv2D_()
flatten = _Flatten_()
CrossEntropyLoss = _CrossEntropyLoss_()
MaxPool2d = _MaxPool2D_()
AvgPool2d = _AvgPool2D_()
