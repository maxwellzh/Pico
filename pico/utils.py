from .base import Tensor
from .nn import Module

import os
import glob
import random
import numpy as np
import pickle
from collections import OrderedDict
from PIL import Image
from typing import Callable, Tuple


def save(obj, path: str):
    if not isinstance(obj, OrderedDict):
        obj = OrderedDict(obj)

    with open(path, 'wb') as fo:
        pickle.dump(obj, fo)


def load(path) -> OrderedDict:
    with open(path, 'rb') as fi:
        return pickle.load(fi)


def transform(mean: float, std: float):
    def _transform(img: np.ndarray) -> Tensor:
        out = (img - mean) / std
        if len(out.shape) == 4:
            return Tensor(out, retrain=False)
        elif len(out.shape) == 3:
            return Tensor(out.reshape(out.shape + (1,)), retrain=False)
        else:
            raise NotImplementedError(
                "Could not deal with shape {}".format(out.shape))

    return _transform


def clip_grad(model: Module, threshold: float = 0.2):
    for param in model.parameters():
        grad_norm = np.linalg.norm(param.grad)
        param_norm = np.linalg.norm(param.data)
        if grad_norm > threshold * param_norm:
            param.grad /= grad_norm / (threshold*param_norm)


class DataLoader(object):
    def __init__(
            self,
            dir: str,
            transform: Callable[[np.ndarray], Tensor],
            batchsize: int,
            shuffle: bool = False,
            img_format: str = 'bmp') -> None:
        super().__init__()
        assert os.path.isdir(dir)
        assert isinstance(img_format, str)
        self.data = []
        for c in [os.path.basename(c) for c in glob.glob(f"{dir}/*") if os.path.isdir(c)]:
            label = int(c)
            self.data += [(_img, label)
                          for _img in glob.glob(f"{dir}/{c}/*.{img_format}")]

        self.transform = transform
        self.batchsize = batchsize
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> None:
        return self()

    def __call__(self) -> Tuple[Tensor, Tensor]:
        if self.shuffle:
            random.shuffle(self.data)

        batch_imgs = []
        batch_tags = []
        for i, (img, tag) in enumerate(self.data):
            img = np.array(Image.open(img)).astype(float)
            batch_imgs.append(img)
            batch_tags.append(tag)

            if i % self.batchsize == self.batchsize - 1:
                ndimgs = np.stack(batch_imgs, axis=0)
                ndtags = np.stack(batch_tags, axis=0)
                batch_imgs = []
                batch_tags = []
                yield self.transform(ndimgs), Tensor(ndtags, retrain=False)

        if batch_imgs != [] and batch_tags != []:
            ndimgs = np.stack(batch_imgs, axis=0)
            ndtags = np.stack(batch_tags, axis=0)
            batch_imgs = []
            batch_tags = []
            yield self.transform(ndimgs), Tensor(ndtags, retrain=False)
