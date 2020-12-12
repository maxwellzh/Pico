import numpy as np
import pico.functional as F
from pico.base import Tensor
from pico.base import tracer
from pico.utils import SGD, Adam

a = Tensor(np.random.randn(1), requires_grad=True)
b = Tensor(np.random.randn(1), requires_grad=True)

print(a)
print(b)

optimizer = Adam([a, b], lr=0.1)

for _ in range(5):

    optimizer.zero_grad()
    d = a + b

    f = a * d

    f.backward()

    optimizer.step()
    print("")

    print(a, a.grad)
    print(b, b.grad)