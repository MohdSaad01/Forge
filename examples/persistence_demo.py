"""Milestone 7 manual verification: train -> save -> [process exits] -> load -> predict.

Run directly: `python examples/persistence_demo.py`

Phase 1 (this process) builds a tiny `Linear` model, trains it with the
existing `Trainer` (Milestone 6) on a deterministic synthetic regression
dataset, records a prediction, and saves the trained model with
`forge.save_model()`.

Phase 2 launches a **separate Python process** (`--load-and-predict`) that
only imports Forge and calls `forge.load_model()` on the saved file -- it
never touches the original `model` object, `Trainer`, or dataset. This is
the strongest available proof that the saved file, not any leftover Python
state, is what makes inference work: the loading process starts cold.

The script asserts the two processes' predictions are numerically
equivalent before printing a final PASS/FAIL summary.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "_persistence_demo_model.forge"
QUERY_PATH = Path(__file__).parent / "_persistence_demo_query.npy"


def train_and_save() -> np.ndarray:
    import forge
    from forge import Tensor, no_grad
    from forge.data import DataLoader, TensorDataset
    from forge.nn import Linear
    from forge.nn.loss import MSELoss
    from forge.optim import SGD
    from forge.training import Trainer

    print("=" * 70)
    print("Phase 1 (this process): train a Linear model, y = 3*x1 - 2*x2 + 1")
    print("=" * 70)

    forge.random.seed(0)
    rng = np.random.default_rng(0)

    n_samples = 200
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=np.random.default_rng(1))

    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)

    history = trainer.fit(loader, epochs=25)
    print(f"train loss: {history[0].train_loss:.4f} -> {history[-1].train_loss:.4f}")

    query = rng.uniform(-1, 1, size=(4, 2))
    with no_grad():
        prediction = model(Tensor(query)).numpy()
    print(f"prediction before save:\n{prediction}")

    forge.save_model(model, str(MODEL_PATH))
    print(f"saved model to {MODEL_PATH}")

    np.save(QUERY_PATH, query)
    return prediction


def load_and_predict() -> None:
    """Runs in a fresh process: only loads the saved file and predicts."""
    import forge
    from forge import Tensor, no_grad

    model = forge.load_model(str(MODEL_PATH))
    query = np.load(QUERY_PATH)
    with no_grad():
        prediction = model(Tensor(query)).numpy()
    # Print as the sole stdout line so the parent process can parse it.
    print(",".join(repr(float(v)) for v in prediction.ravel()))


def main() -> None:
    try:
        pre_save_prediction = train_and_save()

        print("\n" + "=" * 70)
        print("Phase 2 (fresh subprocess): load the saved file cold, predict")
        print("=" * 70)
        result = subprocess.run(
            [sys.executable, str(Path(__file__)), "--load-and-predict"],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout.strip())
        if result.stderr.strip():
            print("[subprocess stderr]\n" + result.stderr, file=sys.stderr)

        post_load_prediction = np.array(
            [float(v) for v in result.stdout.strip().splitlines()[-1].split(",")]
        )

        print(f"\npre-save prediction:  {pre_save_prediction.ravel()}")
        print(f"post-load prediction: {post_load_prediction}")
        assert np.allclose(pre_save_prediction.ravel(), post_load_prediction, atol=1e-6), (
            "post-load prediction did not match pre-save prediction"
        )
        print("\nPASS: loaded model (in a fresh process) reproduces the original prediction.")
    finally:
        MODEL_PATH.unlink(missing_ok=True)
        QUERY_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    if "--load-and-predict" in sys.argv:
        load_and_predict()
    else:
        main()
