"""
title : utils.py
create : @tarickali 23/12/07
update: @tarickali 23/12/07
"""

import numpy as np
from core.types import Shape


def extend_shape(x: np.ndarray, shape: Shape) -> np.ndarray:
    """Extend the shape of an array to a broadcastable array.

    Parameters
    ----------
    x : np.ndarray
        The array to be extended
    shape : Shape
        The shape x will be extended to

    Returns
    -------
    np.ndarray

    """

    a = x
    if x.shape != shape:
        if x.size == np.prod(shape):
            a = np.reshape(x, shape)
        else:
            a = np.array(np.broadcast_to(x, shape))
    return a


def reduce_shape(x: np.ndarray, shape: Shape) -> np.ndarray:
    """Reduce the shape of an array from a broadcastable array.

    Parameters
    ----------
    x : np.ndarray
        The array to be reduced
    shape : Shape
        The shape x will be reduced to

    Returns
    -------
    np.ndarray

    """

    a = x
    if x.shape != shape:
        if x.size == np.prod(shape):
            a = np.reshape(x, shape)
        else:
            if len(shape) < 1:
                a = np.mean(x).reshape(shape)
            else:
                for i, ax in enumerate(np.broadcast_shapes(x.shape, shape)):
                    if shape[i] != ax:
                        break
                a = np.mean(x, axis=i).reshape(shape)
    return a
