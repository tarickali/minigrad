"""
title : main.py
create : @tarickali 23/12/07
update : @tarickali 23/12/07
"""

from core import Tensor
from core.functions import *


def main():
    x = Tensor(1)
    y = Tensor([[2, 2, 1]])

    t = tanh(x)

    print(t)

    t.backward()

    print(t.grad, x.grad)


if __name__ == "__main__":
    main()
