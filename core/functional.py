"""
title : functional.py
create : @tarickali 23/12/07
update : @tarickali 23/12/09
"""

import numpy as np

from core.tensor import Tensor

__all__ = ["identity", "sigmoid", "relu", "tanh", "log", "clip", "softmax"]


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


def log(x: Tensor) -> Tensor:
    data = np.log(np.maximum(x.data, 1e-15))
    output = Tensor(data=data, children=(x,))

    def backward():
        x.grad += (1 / np.maximum(x.data, 1e-15)) * output.grad

    output._backward = backward

    return output


def clip(x: Tensor, low: float, high: float) -> Tensor:
    """Clip tensor values to [low, high]. Gradient passes through where low < x < high."""
    data = np.clip(x.data, low, high)
    output = Tensor(data=data, children=(x,))

    def backward():
        mask = (x.data >= low) & (x.data <= high)
        x.grad += mask * output.grad

    output._backward = backward

    return output


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    # Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x.data - x_max)
    data = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    output = Tensor(data=data, children=(x,))

    def backward():
        # d(softmax)/dx = s * (dL/ds - sum(s * dL/ds))
        s = data
        grad_sum = np.sum(s * output.grad, axis=axis, keepdims=True)
        x.grad += s * (output.grad - grad_sum)

    output._backward = backward

    return output
