import pico
import numpy as np
import pico.functional as F
from pico.base import Tensor
from pico.base import tracer

a = Tensor(np.random.randn(1), requires_grad=True)
b = Tensor(np.random.randn(1), requires_grad=True)

c = a - b
d = b - a
f = c * d

f.backward()
# print(tracer.tensors)
# print(tracer.funcs)
# print('')

print(f, f.grad)
print(d, d.grad)
print(c, c.grad)
print(b, b.grad)
print(a, a.grad)
