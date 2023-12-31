"""
title : tensor.py
create : @tarickali 23/12/07
update : @tarickali 23/12/10
"""

from __future__ import annotations
import numpy as np

from core.types import *
from utils import extend_shape, reduce_shape


class Tensor:
    def __init__(
        self,
        data: Array | Numeric,
        dtype: Dtype = None,
        children: tuple[Tensor, ...] = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data if dtype == None else data.astype(dtype)
        self.grad = np.zeros_like(self.data, dtype=np.float32)

        # Info used for flows in the computation graph
        # self.parents = () if parents == None else parents
        self.children = () if children == None else children

        # Info used for computation
        # self._forward = lambda : None
        self._backward = lambda: None

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.grad)

    def backward(self) -> None:
        order = list[Tensor]()
        visited = set[Tensor]()

        def build(x: Tensor) -> None:
            if x not in visited:
                visited.add(x)
                for child in x.children:
                    build(child)
                order.append(x)

        build(self)

        self.grad = np.ones_like(self.data)
        for x in reversed(order):
            x._backward()

    def __add__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)

        output = Tensor(data=self.data + other.data, children=(self, other))

        def backward():
            self.grad = extend_shape(self.grad, output.grad.shape)
            other.grad = extend_shape(other.grad, output.grad.shape)
            self.grad += output.grad
            other.grad += output.grad
            self.grad = reduce_shape(self.grad, self.data.shape)
            other.grad = reduce_shape(other.grad, other.data.shape)

        output._backward = backward

        return output

    def __sub__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)

        output = Tensor(data=self.data - other.data, children=(self, other))

        def backward():
            self.grad = extend_shape(self.grad, output.grad.shape)
            other.grad = extend_shape(other.grad, output.grad.shape)
            self.grad += output.grad
            other.grad += -output.grad
            self.grad = reduce_shape(self.grad, self.data.shape)
            other.grad = reduce_shape(other.grad, other.data.shape)

        output._backward = backward

        return output

    def __mul__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)

        output = Tensor(data=self.data * other.data, children=(self, other))

        def backward():
            self.grad = extend_shape(self.grad, output.grad.shape)
            other.grad = extend_shape(other.grad, output.grad.shape)
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
            self.grad = reduce_shape(self.grad, self.data.shape)
            other.grad = reduce_shape(other.grad, other.data.shape)

        output._backward = backward

        return output

    def __matmul__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)

        output = Tensor(data=self.data @ other.data, children=(self, other))

        def backward():
            self.grad = extend_shape(self.grad, output.grad.shape)
            other.grad = extend_shape(other.grad, output.grad.shape)
            self.grad += output.grad @ other.data.T
            other.grad += self.data.T @ output.grad
            self.grad = reduce_shape(self.grad, self.data.shape)
            other.grad = reduce_shape(other.grad, other.data.shape)

        output._backward = backward

        return output

    def __pow__(self, other: Numeric) -> Tensor:
        if not isinstance(other, Numeric):
            raise ValueError("Cannot perform operation on non numeric value.")

        output = Tensor(data=self.data**other, children=(self,))

        def backward():
            self.grad += (other * self.data ** (other - 1)) * output.grad

        output._backward = backward

        return output

    def __neg__(self) -> Tensor:
        output = Tensor(data=-self.data, children=(self,))

        def backward():
            self.grad += -output.grad

        output._backward = backward

        return output

    def __truediv__(self, other: Tensor | Array | Numeric) -> Tensor:
        return self * other**-1

    def __radd__(self, other: Tensor | Array | Numeric) -> Tensor:
        return self + other

    def __rsub__(self, other: Tensor | Array | Numeric) -> Tensor:
        return -self + other

    def __rmul__(self, other: Tensor | Array | Numeric) -> Tensor:
        return self * other

    # TODO: add rmatmul?

    def __eq__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)
        return Tensor(data=self.data == other.data)

    def __ne__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)
        return Tensor(data=self.data != other.data)

    def __ge__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)
        return Tensor(data=self.data >= other.data)

    def __gt__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)
        return Tensor(data=self.data > other.data)

    def __le__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)
        return Tensor(data=self.data <= other.data)

    def __lt__(self, other: Tensor | Array | Numeric) -> Tensor:
        if not isinstance(other, Tensor | Array | Numeric):
            raise ValueError(f"Cannot perform Tensor operation on {type(other)}.")
        if isinstance(other, Array | Numeric):
            other = Tensor(other)
        return Tensor(data=self.data < other.data)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data.__repr__()}, dtype={self.dtype}, shape={self.shape})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> Dtype:
        return self.data.dtype
