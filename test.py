import numpy as np
import pico.functional as F
from pico.base import Tensor
import pico.module as nn
import pico.utils as utils
import pico.optimizer as optim
import matplotlib.pyplot as plt

# trainloader = DataLoader('data/MNIST/train', transform(0.1307, 0.3081), 2048, True)

# for data, label in trainloader():
#     print(data.size(), label.size())

# exit(0)

batchsize = 256
dim_hid = 6
x = 5*np.random.randn(batchsize, dim_hid)
W0 = 3*np.random.randn(dim_hid, 1)
y_label = x @ W0 + np.random.randn(batchsize, 1)*0.2

X = Tensor(x)
label = Tensor(y_label)

model = nn.Sequential([
    nn.Linear(dim_hid, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
])

optimizer = optim.Adam(model.parameters(), lr=0.01)

# ckpt = utils.load('./linear.pt')
# model.load_state_dict(ckpt['model'])
# optimizer.load_state_dict(ckpt['optimizer'])
# print(model.weights)
# print(model.bias)
# exit(0)
log = []
for _ in range(300):

    optimizer.zero_grad()
    # d = a + b

    out = model(X)

    loss = F.EuclidLoss(out, label)

    loss.backward(batchsize)
    # print(model.bias.grad)

    # print(loss.data)
    log.append(loss.data)

    optimizer.step()


for name, param in model.named_parameters():
    print(name, param.data)

print(W0)
# print(W1)
# print(W2)
plt.semilogy(log)
plt.grid(ls='--')
plt.savefig('loss.jpg', dpi=300)
utils.save([('model', model.state_dict()),
            ('optimizer', optimizer.state_dict())], './linear.pt')
