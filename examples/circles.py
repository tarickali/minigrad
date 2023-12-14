"""
title : circles.py
create : @tarickali 23/12/13
update : @tarickali 23/12/13
"""

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from core import Tensor
from mlp import MLP
from core.functional import relu
import core.math as cm

__all__ = ["circles_driver"]


def train(
    model: MLP,
    X: Tensor,
    y: Tensor,
    alpha: float = 0.1,
    epochs: int = 300,
) -> list:
    history = []
    for e in range(epochs):
        o = model(X)

        # SVM max-margin loss
        losses = relu(-y * o + 1)
        loss = cm.sum(losses) / losses.shape[0]

        model.zero_grad()
        loss.backward()

        for param in model.parameters():
            param.data -= alpha * param.grad

        accuracy = (y > 0) == (o > 0)
        accuracy = cm.sum(accuracy) / accuracy.shape[0]
        history.append({"epoch": e, "loss": loss.data, "accuracy": accuracy.data})
        print(history[-1])

    return history


def circles_driver():
    # Generate data
    X, y = make_circles(256, random_state=42)
    y = y.reshape(-1, 1)
    X = Tensor(X)
    y = Tensor(y)

    # Network and training hyperparameters
    EPOCHS = 5
    ALPHA = 0.3

    model = MLP([2, 16, 16, 16, 1])

    # Train network
    history = train(model, X, y, alpha=ALPHA, epochs=EPOCHS)

    # Plot loss curve
    losses = [epoch["loss"] for epoch in history]
    plt.plot(losses)
    plt.show()
