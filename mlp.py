"""
title : mlp.py
create : @tarickali 23/12/07
update : @tarickali 23/12/07
"""

from abc import ABC, abstractmethod

import numpy as np

from core import Tensor
from core.functional import relu


class Module(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Layer(Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.weights = Tensor(np.random.randn(in_dim, out_dim))
        self.bias = Tensor(np.zeros(out_dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weights, self.bias]


class MLP(Module):
    def __init__(self, dims: list[int]) -> None:
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.layers) - 1):
            x = relu(self.layers[i](x))
        return self.layers[-1](x)

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]
