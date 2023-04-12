import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x) / u

def cross_entropy_error(y, t):
    # log 0回避用の微小な値を作成
    delta = 1e-7
    return - np.sum(t * np.log(y + delta))