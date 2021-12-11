import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class Perceptron:
    # initialize hyperparameters (learning rate and number of iterations)
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1 + X.shape[1])]  # randomly initialize weights
        self.errors_ = []  # keeps tracks of the number of errors per iteration for observation purposes

        # iterate over labelled dataset updating weights for each features accordingly
        for _ in range(self.n_iter):
            errors = 0
            for xi, label in zip(X, y):
                update = self.eta * (label - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # predict a classification for a sample of features X
    def predict(self, X):
        decision = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(decision >= 0.0, 1, -1)
