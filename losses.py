"""
title : losses.py
create : @tarickali 23/12/09
update : @tarickali 23/12/13
"""

import numpy as np

from core import Tensor
from core.functional import sigmoid, softmax


def mse(y: Tensor, o: Tensor) -> Tensor:
    """Computes the mean squared error between true and pred arrays.

    Parameters
    ----------
    y : Tensor @ (m, 1)
        The target ground truth labels.
    o : Tensor @ (m, 1)
        The network output predictions.

    Returns
    -------
    Tensor : loss @ ()

    """

    assert y.shape == o.shape

    m, _ = y.shape

    loss = sum((y - o) ** 2) / (2 * m)
    assert loss.shape == ()

    return loss


def binary_crossentropy(y: Tensor, o: Tensor, logits: bool = True) -> Tensor:
    """Computes the binary crossentropy between true and pred arrays.

    Parameters
    ----------
    y : Tensor @ (m, 1)
        The target ground truth labels.
    o : Tensor @ (m, 1)
        The network output predictions.
    logits : bool = True
        Indicates whether the values of o are activated.

    Returns
    -------
    Tensor : loss @ ()

    """

    assert y.shape == o.shape

    m, _ = y.shape

    if logits == True:
        a = sigmoid(o)
        loss = sum(-y * np.log(a) - (1 - y) * np.log(1 - a)) / m
    else:
        loss = sum(-y.data * np.log(o.data) - (1 - y.data) * np.log(1 - o.data)) / m

    assert loss.shape == ()

    return loss


def categorical_crossentropy(y: Tensor, o: Tensor, logits: bool = True) -> Tensor:
    """Computes the categorical crossentropy between true and pred arrays.

    Parameters
    ----------
    y : Tensor @ (m, n_out)
        The target ground truth labels.
    o : Tensor @ (m, n_out)
        The network output predictions.
    logits : bool = True
        Indicates whether the values of o are activated.

    Returns
    -------
    Tensor : loss @ ()


    """

    assert y.shape == o.shape

    if logits == True:
        a = softmax(o)
        loss = -np.mean(y.data * np.log(a.data))
    else:
        loss = -np.mean(y.data * np.log(o.data))
    assert loss.shape == ()

    return loss
