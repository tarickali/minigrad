"""
title : losses.py
create : @tarickali 23/12/09
update : @tarickali 23/12/13
"""

import numpy as np

from core import Tensor
from core.functional import sigmoid, softmax, log, clip
from core import math as cm


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

    loss = cm.sum((y - o) ** 2) / (2 * m)
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

    if logits is True:
        a = sigmoid(o)
        one = Tensor(np.ones_like(a.data))
        loss = cm.sum(-y * log(a) - (1 - y) * log(clip(one - a, 1e-15, 1.0))) / m
    else:
        # o is already probabilities; clip for numerical stability and keep graph
        a = clip(o, 1e-15, 1 - 1e-15)
        one = Tensor(np.ones_like(a.data))
        loss = cm.sum(-y * log(a) - (1 - y) * log(clip(one - a, 1e-15, 1.0))) / m

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

    n = y.shape[0]
    if logits is True:
        a = softmax(o)
        loss = cm.sum(-y * log(a)) / n
    else:
        a = clip(o, 1e-15, 1 - 1e-15)
        loss = cm.sum(-y * log(a)) / n
    assert loss.shape == ()

    return loss
