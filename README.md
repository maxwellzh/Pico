![logo](./assets/logo.png) 

# Pico

**Pico** is a numpy-based "pico"  neural network framework, with torch-like coding style and auto-grad implementation.

- [x] Flexible Tensor
- [x] Auto-grad mechanism
- [x] Compact codes
- [ ] ~~High performace and efficiency~~

## Requirements

Install requirements of `pico` via:

```bash
pip install -r requirements.txt
```

Preparing the MNIST images further requires to install `torch` and `torchvision`, but they're not required for the basic functions.

## Usage

The coding style of **Pico** is just almost the same as pytorch. See `mnist.py` for MNIST classification example.

Here is a mini-example:

```python
>>> import pico
>>> import pico.functional as F
>>> import numpy as np

# create a Tensor, initialized with a numpy array, set requires_grad=True to calculate gradient
>>> x = pico.Tensor(np.random.randn(7, 5), requires_grad=True)
# create the target data
>>> target = pico.Tensor(np.random.randint(0, 5, (7,)))
# compute cross entropy loss
>>> loss = F.CrossEntropyLoss(x, target)
>>> loss
Tensor(1.7836267107242398, requires_grad=True)
# call backward of loss
>>> loss.backward()
>>> print(x.grad)
[[ 0.18704112  0.07444789  0.10220077  0.07920804  0.0466561 ]
 [ 0.0495909   0.0595846   0.36349058  0.3803355   0.04203391]
 [ 0.59336148 -0.05008445  0.1262403   0.04342903  0.1949402 ]
 [ 0.16987114  0.21257814  0.00249838  0.10804468  0.35199102]
 [ 0.09629747  0.03482448  0.43659541  0.20108366  0.08404939]
 [-0.1193856   0.02339847  0.09020922  0.21635362  0.16244956]
 [ 0.24815381  0.04523672  0.0740008   0.17944171  0.05384478]]
```

## MNIST

**Data prepare:** Firstly install `torch` module for loading data.

```bash
pip install torch==1.10.2 torchvision==0.11.3
```

Then run command

```bash
python prepare_data.py
```

Waiting the script to finish.

**Training:**

```bash
python3 mnist.py
```

## Pico module file list

```
pico
├── __init__.py
├── base.py
├── functional.py
├── nn.py
├── optimizer.py
└── utils.py
```

