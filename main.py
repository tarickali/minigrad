"""
title : main.py
create : @tarickali 23/12/07
update : @tarickali 23/12/07
"""

import numpy as np
from core import Tensor
from core.functions import *
from mlp import MLP


def main():
    net = MLP([784, 512, 512, 10])
    x = Tensor(np.random.randn(1, 784))

    y = net(x)
    print(y.grad)
    print(x.grad)

    y.backward()

    print(y.grad)
    print(x.grad)


if __name__ == "__main__":
    main()
