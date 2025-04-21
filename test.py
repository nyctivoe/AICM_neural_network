from re import X
import util
from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def check(n, Xs, Ys, NN, verbose=False):
    X_test = Xs[:n]
    y_test = Ys[:n]
    
    predictions = NN.predict(X_test)
    
    if y_test.ndim > 1:
        actual = np.argmax(y_test, axis=1)
    else:
        actual = y_test
    
    # output incorrect predictions
    
    incorrect_indices = np.where(predictions != actual)[0]
    if len(incorrect_indices) > 0:
        print("Incorrect predictions at indices:", incorrect_indices)
        if verbose:
            for i in incorrect_indices:
                print(f"Predicted: {predictions[i]}, Actual: {actual[i]}")
                util.mnist_visualize((Xs, Ys), index=i, num_images=1, figsize=(2, 2), title_fmt="Pred: {}")
    
    correct = np.sum(predictions == actual)
    print(f"Accuracy: {correct}/{n} ({correct/n*100:.2f}%)")
    print(f"Predictions: {predictions}")
    print(f"Actual: {actual}")

Xs, Ys = util.read_mnist_csv("mnist_train.csv")
Xt, Yt = util.read_mnist_csv("mnist_test.csv")

network = NeuralNetwork([784, 128, 128, 10], "prelu", seed=0)

print("Pretrain - checking...")
check(100, Xs, Ys, network)

print("Training...")
network.train(Xs, Ys, epochs=20, verbose=True)

print("Posttrain - checking...")
check(10000, Xs, Ys, network, verbose=False)