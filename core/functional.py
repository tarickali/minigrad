"""
title : functional.py
create : @tarickali 23/12/07
udpate : @tarickali 23/12/09
"""

import numpy as np

from core.tensor import Tensor

__all__ = ["identity", "sigmoid", "relu", "tanh"]


def identity(x: Tensor) -> Tensor:
    data = x.data
    output = Tensor(data=data, children=(x,))

    def backward():
        x.grad += output.grad

    output._backward = backward

    return output


def sigmoid(x: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-x.data))
    output = Tensor(data=data, children=(x,))

    def backward():
        x.grad += ((1 - data) * data) * output.grad

    output._backward = backward

    return output


def relu(x: Tensor) -> Tensor:
    data = np.maximum(x.data, 0)
    output = Tensor(data=data, children=(x,))

    def backward():
        x.grad += (output.data > 0) * output.grad

    output._backward = backward

    return output


def tanh(x: Tensor) -> Tensor:
    data = np.tanh(x.data)
    output = Tensor(data=data, children=(x,))

    def backward():
        x.grad += (1 - data**2) * output.grad

    output._backward = backward

    return output


def softmax(x: Tensor) -> Tensor:
    den = np.sum(x.data)
    data = x.data / den
    output = Tensor(data=data, children=(x,))

    def backward():
        x.grad += np.ones_like(x.data) * output.grad

    output._backward = backward

    return output
