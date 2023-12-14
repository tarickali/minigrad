"""
title : math.py
create : @tarickali 23/12/09
update : @tarickali 23/12/09
"""

import numpy as np
from core.tensor import Tensor
from utils import extend_shape, reduce_shape


def sum(x: Tensor, axis: int | list[int] = None) -> Tensor:
    output = Tensor(data=np.sum(x.data, axis=axis), children=(x,))

    def backward():
        x.grad = extend_shape(x.grad, output.grad.shape)
        x.grad += output.grad
        x.grad = reduce_shape(x.grad, x.data.shape)

    output._backward = backward

    return output
