from turtle import forward
import numpy as np # heheheha
import matplotlib.pyplot as plt # for plotting
import pandas as pd # reading csv files

def read_mnist_csv(filepath):
    df = pd.read_csv(filepath)
    
    y = df.iloc[:, 0].values
    Y = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1.0
    X = df.iloc[:, 1:].values
    
    X = X / 255.0
    
    return X, Y

def mnist_visualize(data, index=0, num_images=5, figsize=(10, 2)):
    X, y = data
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    for i in range(num_images):
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
            
        img_idx = index + i
        if img_idx < len(X):
            img = X[img_idx].reshape(28, 28)
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Label: {y[img_idx]}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def initialize_parameters():
    np.random.seed(42)

    input_size = 784  
    hidden_size1 = 20
    hidden_size2 = 20
    output_size = 10

    W1 = np.random.rand(input_size, hidden_size1) * 0.01
    b1 = np.zeros((1, hidden_size1))
    
    W2 = np.random.rand(hidden_size1, hidden_size2) * 0.01
    b2 = np.zeros((1, hidden_size2))
    
    W3 = np.random.rand(hidden_size2, output_size) * 0.01
    b3 = np.zeros((1, output_size))
    
    parameters = {"w1": W1, "b1": b1,
                  "w2": W2, "b2": b2,
                  "w3": W3, "b3": b3}
    
    return parameters

def normalize_output(z):
    out = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(out)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def PReLU(z):
    return np.where(z > 0, z, 0.001 * z)

def forward_propagation(X, y, parameters, debug=True):
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
        print("it is:", np.argmax(output, axis=1))
        print("it should be:", y)
    
    cache = {
        "z1": z1, "a1": a1,
        "z2": z2, "a2": a2,
        "z3": z3, "a3": a3,
        "output": output
    }
    
    return output, cache

def backward_propagation(X, y, parameters, cache):
    w1, b1 = parameters["w1"], parameters["b1"]
    w2, b2 = parameters["w2"], parameters["b2"]
    w3, b3 = parameters["w3"], parameters["b3"]
    
    a1, a2 = cache["a1"], cache["a2"]
    output = cache["output"]
    
    m = X.shape[0]
    
    delta3 = output - y
    dw3 = np.dot(a2.T, delta3) / m
    db3 = np.sum(delta3, axis=0, keepdims=True) / m
    
    delta2 = np.dot(delta3, w3.T) * (a2 > 0)
    dw2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0, keepdims=True) / m
    
    delta1 = np.dot(delta2, w2.T) * (a1 > 0)
    dw1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0, keepdims=True) / m
    
    grads = {"dw1": dw1, "db1": db1,
             "dw2": dw2, "db2": db2,
             "dw3": dw3, "db3": db3}
    
    return grads

def cost(y, y_hat): 
    """
    Implements the L2 Cost Function

    Args:
        y (numpy.ndarray): intended labels
        y_hat (numpy.ndarray): network labels

    Returns:
        loss (float): L2 cost function value
    """
    loss = np.mean(np.sum((y - y_hat) ** 2, axis=1)) / 2.0
    return loss

parameters = initialize_parameters()

def train(X, y, parameters, learning_rate=0.01, epochs=1000, batch_size=100):
    m = X.shape[0]
    num_batches = m // batch_size
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle data
        indices = np.random.permutation(m)
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
            epoch_loss += batch_loss
        
        epoch_loss /= num_batches
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    return parameters

def check(n):
    X_test = Xs[:n]
    y_test = ys[:n]
    
    output, cache = forward_propagation(X_test, y_test, parameters, debug=False)
    predictions = np.argmax(output, axis=1)
    
    if y_test.ndim > 1:
        actual = np.argmax(y_test, axis=1)
    else:
        actual = y_test
    
    correct = np.sum(predictions == actual)
    print(f"Accuracy: {correct}/{n} ({correct/n*100:.2f}%)")
    print(f"Predictions: {predictions}")
    print(f"Actual: {actual}")

train_data = read_mnist_csv('mnist_train.csv')
test_data = read_mnist_csv('mnist_test.csv')
Xs, ys = train_data

mnist_visualize(train_data, index=0, num_images=10, figsize=(10, 2))

parameters = initialize_parameters()
parameters = train(Xs, ys, parameters, learning_rate=0.02, epochs=100, batch_size=64)
check(100)