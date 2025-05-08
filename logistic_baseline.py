"""Based off code for HW2 Problem 1"""
import argparse
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid

OPTS = None

def predict(w, X):
    """Return the predictions using weight vector w on inputs X.

    Args:
        - w: Vector of size (D,)
        - X: Matrix of size (M, D)
    Returns:
        - Predicted classes as a numpy vector of size (M,). Each entry should be either -1 or +1.
    """
    preds_sig = sigmoid(X.dot(w)) 
    preds_class = [1 if pred > 0.5 else -1 for pred in preds_sig]
    return preds_class


def train(X_train, y_train, lr=1e-1, num_iters=5000, l2_reg=0.0):
    """Train linear regression using gradient descent.

    Args:
        - X_train: Matrix of size (N, D)
        - y_train: Vector os size (N,)
        - lr: Learning rate
        - num_iters: Number of iterations of gradient descent to run
        - l2_reg: lambda hyperparameter for using L2 regularization
    Returns:
        - Weight vector w of size (D,)
    """
    N, D = X_train.shape
    w = np.zeros(D)
    for t in range(num_iters):
        preds = X_train.dot(w)
        margins = (-1 * y_train) * (preds)

        gradient = (1/N) * -1 * (sigmoid(margins) * y_train).dot(X_train) + (l2_reg * w)
        w -= lr * gradient
    #print("w: ", w)
    return w

def evaluate(w, X, y, name):
    """Measure and print accuracy of a predictor on a dataset."""
    y_preds = predict(w, X)
    acc = np.mean(y_preds == y)
    print('    {} Accuracy: {}'.format(name, acc))
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=2)
    parser.add_argument('--num-iters', '-T', type=int, default=10000)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot-weights')
    return parser.parse_args()

import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from processing import processData

def main():
    # Read the data
    all_data = pd.read_csv("data/water_potability.csv")
    X_train, y_train, X_dev, y_dev, X_test, y_test = processData(all_data)
    y_train = y_train.replace({0: -1, 1: 1})
    y_dev = y_dev.replace({0: -1, 1: 1})
    y_test= y_test.replace({0: -1, 1: 1})

    #print(X_train)

    # Train with gradient descent
    w = train(X_train, y_train, lr=OPTS.learning_rate, num_iters=OPTS.num_iters, l2_reg=OPTS.l2)

    # Evaluate model
    train_acc = evaluate(w, X_train, y_train, 'Train')
    dev_acc = evaluate(w, X_dev, y_dev, 'Dev')
    if OPTS.test:
        test_acc = evaluate(w, X_test, y_test, 'Test')

if __name__ == '__main__':
    OPTS = parse_args()
    main()

