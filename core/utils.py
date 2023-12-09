"""
title : utils.py
create : @tarickali 23/12/07
update: @tarickali 23/12/09
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
            try:
                a = np.array(np.broadcast_to(x, shape))
            except:
                pass
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
                try:
                    broad = np.broadcast_shapes(x.shape, shape)
                except:
                    pass
                else:
                    # Get the intermediate broadcast shape by prepending 1s
                    inter = [1] * (len(broad) - len(shape)) + list(shape)
                    # Get the axis indices that are not the same
                    axes = []
                    for i in range(len(broad) - 1, -1, -1):
                        if x.shape[i] != inter[i]:
                            axes.append(i)
                    a = np.mean(x, axis=tuple(axes)).reshape(shape)
    return a
