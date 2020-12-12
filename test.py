import numpy as np
import pico.functional as F
from pico.base import Tensor
from pico.base import tracer
from pico.utils import SGD, Adam
from pico.module import Linear

batchsize = 256
dim_hid = 6
x = 5*np.random.randn(batchsize, dim_hid)
W = 3*np.random.randn(dim_hid, 1)
y_label = np.matmul(x, W) + np.random.randn(batchsize, 1)*0.2

X = Tensor(x)
label = Tensor(y_label)

model = Linear(dim_hid, 1)

optimizer = Adam(model.parameters(), lr=1)

for _ in range(300):

    optimizer.zero_grad()
    # d = a + b

    out = model(X)

    loss = F.euclidloss(out, label)

    loss.backward(batchsize)
    # print(model.bias.grad)

    print(loss.data)

    optimizer.step()

print(model.weights)
print(model.bias)
print(W)