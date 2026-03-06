# Minigrad

A minimal autograd engine and neural network library built from scratch in Python and NumPy. Minigrad implements reverse-mode automatic differentiation with a dynamic computation graph, enabling gradient-based optimization for small-scale machine learning experiments and educational use.

## Features

- **Differentiable tensor** with NumPy-backed storage and automatic gradient tracking
- **Core ops**: add, subtract, multiply, matrix multiply, power, negation, division
- **Activations**: ReLU, sigmoid, tanh, softmax (numerically stable), plus `log` and `clip`
- **Reductions**: `sum` with optional axis for backprop-through-reduce
- **Losses**: MSE, binary cross-entropy, categorical cross-entropy (all differentiable)
- **Modules**: `Layer` (linear) and `MLP` (multi-layer perceptron) with a simple `Module` base
- **Utilities**: shape broadcast helpers, one-hot encoding, batching for training

## Requirements

- Python 3.9 or later (3.10+ recommended for type-union checks in operations)
- NumPy  
- Optional (for examples): scikit-learn, matplotlib

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/tarickali/minigrad.git
cd minigrad
pip install -r requirements.txt
```

## Quick Start

```python
from core import Tensor
from core.functional import relu, sigmoid
import core.math as math
from mlp import MLP

# Build a small MLP
model = MLP([2, 16, 16, 1])
x = Tensor([[1.0, -1.0], [0.5, 0.5]])

# Forward pass
out = model(x)

# Backward pass (e.g. after a loss)
loss = math.sum(out) / out.shape[0]
model.zero_grad()
loss.backward()

# Gradient descent step
for p in model.parameters():
    p.data -= 0.01 * p.grad
```

## Project Structure

```
minigrad/
├── core/
│   ├── __init__.py    # Exposes Tensor and types
│   ├── tensor.py      # Tensor class and operator overloading
│   ├── types.py       # Type aliases (Array, Dtype, Shape, etc.)
│   ├── functional.py  # identity, sigmoid, relu, tanh, log, clip, softmax
│   └── math.py        # sum (with axis)
├── mlp.py             # Module, Layer, MLP
├── losses.py          # mse, binary_crossentropy, categorical_crossentropy
├── utils.py           # extend_shape, reduce_shape, one_hot, get_batches
├── main.py            # Entry point (runs examples/circles)
├── examples/
│   └── circles.py     # Binary classification on make_circles
├── requirements.txt
├── LICENSE
└── README.md
```

## API Overview

### Tensor

- **Construction**: `Tensor(data)` where `data` is a number, list, or NumPy array.
- **Attributes**: `.data`, `.grad`, `.shape`, `.dtype`.
- **Methods**: `.zero_grad()`, `.backward()` to run backprop.
- **Operators**: `+`, `-`, `*`, `@` (matmul), `**`, `/`, unary `-`, and comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`). Right-hand variants (`__radd__`, etc.) are supported where applicable.

All operations participate in the computation graph so gradients flow correctly when `.backward()` is called on a scalar loss.

### Modules

- **`Module`**: Abstract base with `forward(x)`, `parameters()`, `zero_grad()`, and `__call__`.
- **`Layer(in_dim, out_dim)`**: Linear layer `x @ W + b` with trainable `weights` and `bias`.
- **`MLP(dims)`**: Stack of linear layers with ReLU between them and no activation on the last layer.

### Losses

- **`mse(y, o)`**: Mean squared error; `y` and `o` same shape.
- **`binary_crossentropy(y, o, logits=True)`**: BCE; set `logits=False` if `o` is already probabilities.
- **`categorical_crossentropy(y, o, logits=True)`**: CCE for one-hot `y` and logits or probabilities `o`.

Each returns a scalar `Tensor` suitable for `.backward()`.

### Running the Example

The circles example trains an MLP on scikit-learn’s “make_circles” dataset and plots the loss curve:

```bash
python main.py
```

Or run the example module directly:

```bash
python -m examples.circles
```

## License

MIT License. See [LICENSE](LICENSE) for details.
