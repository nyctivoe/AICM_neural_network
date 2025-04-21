import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cupy as cp  # GPU-accelerated array library
import time

# Device selection function
def get_array_module(use_gpu=True):
    """Returns the appropriate array module (NumPy or CuPy) based on GPU availability"""
    if use_gpu:
        try:
            # Test if CUDA is available
            cp.cuda.runtime.getDeviceCount()
            print("Using GPU acceleration with CuPy")
            return cp
        except:
            print("GPU not available, falling back to CPU")
            return np
    return np

# Data transfer utilities
def to_gpu(x):
    """Transfer NumPy array to GPU"""
    if isinstance(x, np.ndarray):
        return cp.asarray(x)
    return x

def to_cpu(x):
    """Transfer CuPy array to CPU"""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x

def read_mnist_csv(filepath, use_gpu=True):
    xp = get_array_module(use_gpu)
    df = pd.read_csv(filepath)
    
    y = df.iloc[:, 0].values
    Y = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1.0
    X = df.iloc[:, 1:].values
    
    X = X / 255.0
    
    # Transfer to GPU if available
    if use_gpu and xp is cp:
        X = to_gpu(X)
        Y = to_gpu(Y)
    
    return X, Y

def mnist_visualize(
        data,
        index: int = 0,
        num_images: int = 10,
        cols: int = 5,
        figsize: tuple | None = None,
        titles: list[str] | None = None,
        cmap: str = "gray",
        title_fmt: str = "Label: {}",
        title_fontsize: int = 10,
        h_pad: float = 1.0,
        w_pad: float = 0.3,
):
    X, y = data
    # Convert arrays to CPU for visualization
    X = to_cpu(X)
    y = to_cpu(y)
    
    rows = int(np.ceil(num_images / cols))

    if figsize is None:
        figsize = (2.2 * cols, 2.4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(num_images):
        img_idx = index + i
        ax = axes[i]

        if img_idx < len(X):
            img = X[img_idx].reshape(28, 28)
            ax.imshow(img, cmap=cmap)
            
            if titles is not None:
                title = titles[i]
            else:
                title = title_fmt.format(y[img_idx])

            ax.set_title(title, fontsize=title_fontsize, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(num_images, rows * cols):
        axes[j].axis("off")

    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    plt.show()

def initialize_parameters(use_gpu=True):
    xp = get_array_module(use_gpu)
    np.random.seed(42)
    
    input_size = 784  
    hidden_size1 = 100
    hidden_size2 = 20
    output_size = 10

    # Initialize on CPU, then transfer to GPU if needed
    W1 = np.random.rand(input_size, hidden_size1) * 0.01
    b1 = np.zeros((1, hidden_size1))
    
    W2 = np.random.rand(hidden_size1, hidden_size2) * 0.01
    b2 = np.zeros((1, hidden_size2))
    
    W3 = np.random.rand(hidden_size2, output_size) * 0.01
    b3 = np.zeros((1, output_size))
    
    if use_gpu and xp is cp:
        W1, b1 = to_gpu(W1), to_gpu(b1)
        W2, b2 = to_gpu(W2), to_gpu(b2)
        W3, b3 = to_gpu(W3), to_gpu(b3)
    
    parameters = {"w1": W1, "b1": b1,
                  "w2": W2, "b2": b2,
                  "w3": W3, "b3": b3}
    
    return parameters

def normalize_output(z):
    xp = cp if isinstance(z, cp.ndarray) else np
    out = z - xp.max(z, axis=1, keepdims=True)
    exp_z = xp.exp(out)
    return exp_z / xp.sum(exp_z, axis=1, keepdims=True)

def PReLU(z):
    xp = cp if isinstance(z, cp.ndarray) else np
    return xp.where(z > 0, z, 0.001 * z)

def forward_propagation(X, y, parameters, debug=True):
    xp = cp if isinstance(X, cp.ndarray) else np
    
    w1, b1 = parameters["w1"], parameters["b1"]
    w2, b2 = parameters["w2"], parameters["b2"]
    w3, b3 = parameters["w3"], parameters["b3"]

    z1 = X @ w1 + b1
    a1 = PReLU(z1)

    z2 = a1 @ w2 + b2
    a2 = PReLU(z2)

    z3 = a2 @ w3 + b3
    a3 = PReLU(z3)
    
    output = normalize_output(a3)
    
    if debug:
        # Convert to CPU for printing if needed
        output_cpu = to_cpu(output)
        y_cpu = to_cpu(y)
        print("it is:", np.argmax(output_cpu, axis=1))
        print("it should be:", y_cpu)
    
    cache = {
        "z1": z1, "a1": a1,
        "z2": z2, "a2": a2,
        "z3": z3, "a3": a3,
        "output": output
    }
    
    return output, cache

def backward_propagation(X, y, parameters, cache):
    xp = cp if isinstance(X, cp.ndarray) else np
    
    w1, b1 = parameters["w1"], parameters["b1"]
    w2, b2 = parameters["w2"], parameters["b2"]
    w3, b3 = parameters["w3"], parameters["b3"]
    
    a1, a2 = cache["a1"], cache["a2"]
    output = cache["output"]
    
    m = X.shape[0]
    
    delta3 = output - y
    dw3 = xp.dot(a2.T, delta3) / m
    db3 = xp.sum(delta3, axis=0, keepdims=True) / m
    
    delta2 = xp.dot(delta3, w3.T) * (a2 > 0)
    dw2 = xp.dot(a1.T, delta2) / m
    db2 = xp.sum(delta2, axis=0, keepdims=True) / m
    
    delta1 = xp.dot(delta2, w2.T) * (a1 > 0)
    dw1 = xp.dot(X.T, delta1) / m
    db1 = xp.sum(delta1, axis=0, keepdims=True) / m
    
    grads = {"dw1": dw1, "db1": db1,
             "dw2": dw2, "db2": db2,
             "dw3": dw3, "db3": db3}
    
    return grads

def cost(y, y_hat): 
    """Implements the L2 Cost Function"""
    xp = cp if isinstance(y, cp.ndarray) else np
    loss = xp.mean(xp.sum((y - y_hat) ** 2, axis=1)) / 2.0
    return loss

def train(X, y, parameters, learning_rate=0.01, epochs=1000, batch_size=100):
    xp = cp if isinstance(X, cp.ndarray) else np
    m = X.shape[0]
    num_batches = m // batch_size
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle data
        indices = xp.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, m)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            output, cache = forward_propagation(X_batch, y_batch, parameters, debug=False)
            
            # Backward pass
            grads = backward_propagation(X_batch, y_batch, parameters, cache)
            
            # Update parameters
            for key in parameters.keys():
                parameters[key] -= learning_rate * grads["d" + key]
            
            # Calculate loss
            batch_loss = cost(y_batch, output)
            epoch_loss += to_cpu(batch_loss) if isinstance(batch_loss, cp.ndarray) else batch_loss
        
        epoch_loss /= num_batches
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return parameters

def check(n, X, y, parameters):
    xp = cp if isinstance(X, cp.ndarray) else np
    
    X_test = X[:n]
    y_test = y[:n]
    
    output, cache = forward_propagation(X_test, y_test, parameters, debug=False)
    
    # Move to CPU for evaluation
    output_cpu = to_cpu(output)
    y_test_cpu = to_cpu(y_test)
    
    predictions = np.argmax(output_cpu, axis=1)
    
    if y_test_cpu.ndim > 1:
        actual = np.argmax(y_test_cpu, axis=1)
    else:
        actual = y_test_cpu
    
    # Output incorrect predictions
    incorrect_indices = np.where(predictions != actual)[0]
    if len(incorrect_indices) > 0:
        print("Incorrect predictions at indices:", incorrect_indices)
        for i in incorrect_indices[:5]:  # Limit to first 5 incorrects
            print(f"Predicted: {predictions[i]}, Actual: {actual[i]}")
    
    correct = np.sum(predictions == actual)
    print(f"Accuracy: {correct}/{n} ({correct/n*100:.2f}%)")

# Main execution
def main(use_gpu=True):
    xp = get_array_module(use_gpu)
    
    print("Loading data...")
    train_data = read_mnist_csv('mnist_train.csv', use_gpu=use_gpu)
    test_data = read_mnist_csv('mnist_test.csv', use_gpu=use_gpu)
    Xs, ys = train_data

    print("Initializing parameters...")
    parameters = initialize_parameters(use_gpu=use_gpu)

    print("Starting training...")
    parameters = train(Xs, ys, parameters, learning_rate=0.02, epochs=100, batch_size=128)
    
    print("Evaluating model...")
    X_test, y_test = test_data
    check(10000, X_test, y_test, parameters)
    
    return parameters

# Run with GPU acceleration
parameters = main(use_gpu=True)