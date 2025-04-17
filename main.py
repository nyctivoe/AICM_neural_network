"""
Minimal MNIST MLP with **joblib**‑based data‑parallel training.
Copy‑paste into a single .py file, `pip install joblib pandas matplotlib`, and run.
"""

# ─────────────── imports ───────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from joblib import Parallel, delayed      # <─ NEW

# ─────────────── data helpers ───────────────
def read_mnist_csv(filepath: str):
    df = pd.read_csv(filepath)
    y  = df.iloc[:, 0].values
    Y  = np.zeros((y.shape[0], 10))
    Y[np.arange(y.shape[0]), y] = 1.0

    X  = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    return X, Y


def mnist_visualize(
    data, index=0, num_images=10, cols=5,
    figsize=None, titles=None,
    cmap="gray", title_fmt="Label: {}", title_fontsize=10,
    h_pad=1.0, w_pad=0.3,
):
    X, y = data
    rows = int(np.ceil(num_images / cols))
    if figsize is None:
        figsize = (2.2 * cols, 2.4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        img_idx = index + i
        if img_idx < len(X):
            ax.imshow(X[img_idx].reshape(28, 28), cmap=cmap)
            title = titles[i] if titles is not None else title_fmt.format(np.argmax(y[img_idx]))
            ax.set_title(title, fontsize=title_fontsize, pad=6)
        ax.set_xticks([]), ax.set_yticks([])

    for j in range(num_images, rows * cols):
        axes[j].axis("off")

    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    plt.show()


# ─────────────── network core ───────────────
def initialize_parameters(seed=42):
    np.random.seed(seed)
    W1 = np.random.randn(784, 20) * 0.01
    b1 = np.zeros((1, 20))
    W2 = np.random.randn(20, 20) * 0.01
    b2 = np.zeros((1, 20))
    W3 = np.random.randn(20, 10) * 0.01
    b3 = np.zeros((1, 10))
    return {"w1": W1, "b1": b1,
            "w2": W2, "b2": b2,
            "w3": W3, "b3": b3}


def PReLU(z):                 # parametric ReLU with α=0.001
    return np.where(z > 0, z, 0.001 * z)


def normalize_output(z):      # softmax
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def forward_propagation(X, y, params, debug=False):
    a1 = PReLU(X @ params["w1"] + params["b1"])
    a2 = PReLU(a1 @ params["w2"] + params["b2"])
    a3 = PReLU(a2 @ params["w3"] + params["b3"])
    out = normalize_output(a3)

    if debug:
        print("pred :", np.argmax(out, axis=1))
        print("label:", np.argmax(y,  axis=1))

    cache = {"a1": a1, "a2": a2, "output": out}
    return out, cache


def backward_propagation(X, y, params, cache):
    a1, a2, out = cache["a1"], cache["a2"], cache["output"]
    m = X.shape[0]

    δ3   = out - y                            # (m,10)
    dw3  = a2.T @ δ3 / m
    db3  = np.sum(δ3, axis=0, keepdims=True) / m

    δ2   = (δ3 @ params["w3"].T) * (a2 > 0)
    dw2  = a1.T @ δ2 / m
    db2  = np.sum(δ2, axis=0, keepdims=True) / m

    δ1   = (δ2 @ params["w2"].T) * (a1 > 0)
    dw1  = X.T @ δ1 / m
    db1  = np.sum(δ1, axis=0, keepdims=True) / m

    return {"dw1": dw1, "db1": db1,
            "dw2": dw2, "db2": db2,
            "dw3": dw3, "db3": db3}


def cost(y, y_hat):                         # L2 loss
    return 0.5 * np.mean(np.sum((y - y_hat) ** 2, axis=1))


# ─────────────── parallel helpers ───────────────
def _batch_gradients(X_b, y_b, params_snapshot):
    out, cache = forward_propagation(X_b, y_b, params_snapshot, debug=False)
    grads = backward_propagation(X_b, y_b, params_snapshot, cache)
    return grads, cost(y_b, out)


def _aggregate(grads_list):
    agg = {}
    for k in grads_list[0]:
        agg[k] = np.mean([g[k] for g in grads_list], axis=0)
    return agg


# ─────────────── training loop (data‑parallel) ───────────────
def train(
    X, y, params,
    learning_rate=0.02,
    epochs=100,
    batch_size=128,
    n_jobs=-1,                       # ← #CPU cores (‑1 = all)
):
    m = X.shape[0]
    num_batches = (m + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = np.random.permutation(m)
        X, y = X[perm], y[perm]

        # build batch index ranges once per epoch
        bounds = [(i*batch_size, min((i+1)*batch_size, m))
                  for i in range(num_batches)]

        # run all batches in parallel workers
        snapshot = deepcopy(params)       # read‑only in workers
        results  = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_batch_gradients)(X[s:e], y[s:e], snapshot)
            for (s, e) in bounds
        )

        grads_list, losses = zip(*results)
        mean_grads = _aggregate(grads_list)

        for k in params:                  # synchronous SGD update
            params[k] -= learning_rate * mean_grads["d" + k]

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | loss = {np.mean(losses):.4f}")

    return params


# ─────────────── evaluation helper ───────────────
def check(X_test, y_test, params):
    out, _ = forward_propagation(X_test, y_test, params, debug=False)
    preds  = np.argmax(out, axis=1)
    true   = np.argmax(y_test, axis=1)
    acc    = (preds == true).mean()
    print(f"Accuracy: {acc*100:.2f}%  ({preds.sum()} / {len(true)})")


# ─────────────── main script ───────────────
if __name__ == "__main__":
    train_data = read_mnist_csv("mnist_train.csv")
    test_data  = read_mnist_csv("mnist_test.csv")
    X_train, y_train = train_data
    X_test,  y_test  = test_data

    mnist_visualize(train_data, num_images=12, cols=6)

    params = initialize_parameters()
    params = train(
        X_train, y_train, params,
        learning_rate=0.02,
        epochs=100,
        batch_size=64,
        n_jobs=-1                # use all CPU cores
    )

    check(X_test[:1000], y_test[:1000], params)
