import numpy as np

class NeuralNetwork:
    def _prelu(self, z):
        return np.where(z > 0, z, 0.001 * z)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _normalize_output(self, z):
        out = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(out)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def __init__(self, layers: list[int], activation, seed: int = 0):
        np.random.seed(seed)
        
        self.parameters = {}
        self.cache = {}
        self.grads = {}
        self.depth = len(layers)
        
        for i in range(self.depth - 1):
            if layers[i] <= 0 or layers[i + 1] <= 0:
                raise ValueError("Layer sizes must be positive integers.")
            self.parameters[f"W{i + 1}"] = np.random.rand(layers[i], layers[i + 1]) * 0.01
            self.parameters[f"B{i + 1}"] = np.zeros((1, layers[i + 1]))

        if isinstance(activation, str):
            if activation.lower() == "prelu":
                self.activation = self._prelu
            elif self.activation.lower() == "sigmoid":
                self.activation = self._sigmoid
            else:
                raise ValueError(f"Unsupported activation function: {activation}.")
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(f"Activation is neither a string nor a callable: {activation}.")
    
    def forward_propagation(self, X: np.ndarray):
        X = X.copy()
        self.cache[f"A0"] = X
        for i in range(1, self.depth):
            W = self.parameters[f"W{i}"]
            B = self.parameters[f"B{i}"]
            
            Z = X @ W + B
            A = self.activation(Z)
            
            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = A
            
            X = A
        
        self.cache["output"] = self._normalize_output(X)
        self.cache["prediction"] = np.argmax(self.cache["output"], axis=1)
        return self.cache["output"], self.cache["prediction"]
    
    def predict(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        
        output, prediction = self.forward_propagation(X)
        return prediction
    
    def backward_propagation(self, X: np.ndarray, Y: np.ndarray):
        m = X.shape[0]
        Y = Y.copy()
        
        self.grads[f"d{self.depth - 1}"] = self.cache["output"] - Y
        self.grads[f"dW{self.depth - 1}"] = self.cache[f"A{self.depth - 2}"].T @ self.grads[f"d{self.depth - 1}"] / m
        self.grads[f"dB{self.depth - 1}"] = np.sum(self.grads[f"d{self.depth - 1}"], axis=0, keepdims=True) / m
        
        for i in range(self.depth - 2, 0, -1):
            self.grads[f"d{i}"] = self.grads[f"d{i + 1}"] @ self.parameters[f"W{i + 1}"].T * (self.cache[f"Z{i}"] > 0)
            self.grads[f"dW{i}"] = self.cache[f"A{i - 1}"].T @ self.grads[f"d{i}"] / m
            self.grads[f"dB{i}"] = np.sum(self.grads[f"d{i}"], axis=0, keepdims=True) / m
        
        return self.grads
    
    def loss(self, Y: np.ndarray):
        loss = loss = np.mean(np.sum((Y - self.cache["output"]) ** 2, axis=1)) / 2.0
        return loss
        
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.02, batch_size: int = 100, verbose: bool = False):
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("X and Y must be numpy arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must be the same.")
        
        m = X.shape[0]
        num_batches = m // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                self.forward_propagation(X_batch)
                grads = self.backward_propagation(X_batch, Y_batch)
                
                for key in self.parameters.keys():
                    self.parameters[key] -= learning_rate * grads["d" + key]
                
                batch_loss = self.loss(Y_batch)
                epoch_loss += batch_loss
                
                if (verbose):
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {batch + 1}/{num_batches}, Loss: {batch_loss:.4f}")
            
            epoch_loss /= num_batches
        
        return epoch_loss
    
        
    