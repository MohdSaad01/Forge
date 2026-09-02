"""Milestone 15 acceptance test: end-to-end image classification through Trainer.

Builds a small deterministic image-like classification dataset (`(N, 1, H,
W)`, two linearly-separable "bright top half" / "bright bottom half"
classes) and trains

```text
Conv2d(1, 4, kernel_size=3) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear
```

through the existing `Dataset -> DataLoader -> Trainer -> ... -> SGD` flow,
unmodified -- exactly the acceptance criterion in the milestone brief. See
`tests/test_cuda_conv.py::test_cuda_small_cnn_trains_and_reduces_loss` for
the CUDA equivalent.
"""

from __future__ import annotations

import numpy as np

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.nn import Conv2d, Linear, MaxPool2d, Module, ReLU
from forge.nn.loss import CrossEntropyLoss
from forge.optim import SGD
from forge.training import Accuracy, Trainer


class TinyCNN(Module):
    """Conv2d(1,4,k=3) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear, for an 8x8 input.

    Flattening between pooling and `Linear` uses `Tensor.reshape` directly
    (an existing Tensor operation) rather than the `data.transforms.Flatten`
    Transform, since that Transform operates on one un-batched sample at a
    time -- here the batch dimension must be preserved. Either way, no new
    infrastructure is introduced, per the milestone brief.
    """

    def __init__(self):
        super().__init__()
        self.conv = Conv2d(1, 4, kernel_size=3)
        self.relu = ReLU()
        self.pool = MaxPool2d(2)
        # Conv2d(k=3,s=1,p=0) on 8x8 -> 6x6; MaxPool2d(2) -> 3x3 -> flatten to 4*3*3.
        self.fc = Linear(4 * 3 * 3, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        batch = x.shape[0]
        flat = x.reshape(batch, x.shape[1] * x.shape[2] * x.shape[3])
        return self.fc(flat)


def _make_dataset(n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, 8, 8), dtype=np.float32)
    Y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        label = i % 2
        Y[i] = label
        noise = rng.standard_normal((4, 8)).astype(np.float32) * 0.05
        if label == 0:
            X[i, 0, :4, :] = 1.0 + noise
            X[i, 0, 4:, :] = rng.standard_normal((4, 8)).astype(np.float32) * 0.05
        else:
            X[i, 0, 4:, :] = 1.0 + noise
            X[i, 0, :4, :] = rng.standard_normal((4, 8)).astype(np.float32) * 0.05
    return X, Y


def test_conv_relu_maxpool_linear_trains_via_trainer_on_cpu():
    forge.random.seed(0)
    X, Y = _make_dataset(n=40, seed=1)

    dataset = TensorDataset(Tensor(X), Tensor(Y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TinyCNN()
    loss_fn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.2)
    trainer = Trainer(model, loss_fn, opt, device="cpu", metrics=[Accuracy()], verbose=False)

    history = trainer.fit(loader, epochs=15)

    first_loss = history[0].train_loss
    last_loss = history[-1].train_loss
    assert last_loss < first_loss * 0.5

    eval_result = trainer.evaluate(loader)
    assert eval_result.metrics["accuracy"] >= 0.9


def test_conv_model_forward_shape_matches_flatten_arithmetic():
    model = TinyCNN()
    x = Tensor(np.zeros((5, 1, 8, 8), dtype=np.float32))
    y = model(x)
    assert y.shape == (5, 2)


def test_conv_model_all_parameters_receive_gradients_through_trainer_step():
    forge.random.seed(2)
    X, Y = _make_dataset(n=16, seed=3)
    model = TinyCNN()
    loss_fn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.1)

    opt.zero_grad()
    pred = model(Tensor(X))
    loss = loss_fn(pred, Y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} did not receive a gradient"
        assert param.grad.shape == param.shape
