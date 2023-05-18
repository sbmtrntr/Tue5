import numpy as np

class Sigmoid:
    def __init__(self):
        self.y = None
    
    def forward(self, x):
        y = 1.0 / (1.0 + np.exp(-x))
        self.y = y
        return y
    
    def backward(self, dy):
        dy = dy * self.y * (1.0 - self.y)
        return dy

class MSE:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self):
        dy = self.y_true - self.y_pred
        return dy