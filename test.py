import pico
import numpy as np
import pico.functional as F
from pico.base import Tensor
from pico.base import tracer

a = Tensor(np.random.randn(1), requires_grad=True)
b = Tensor(np.random.randn(1), requires_grad=True)

print(tracer.tensors)
print(tracer.funcs)
print('')
c = a + b
print(tracer.tensors)
print(tracer.funcs)
print('')
d = a + c
print(tracer.tensors)
print(tracer.funcs)
print('')
f = c + d
print(tracer.tensors)
print(tracer.funcs)
print('')
f.backward()
print(tracer.tensors)
print(tracer.funcs)
print('')

print(a, a.grad)
print(b, b.grad)
print(c, c.grad)
print(d, d.grad)
print(f, f.grad)
