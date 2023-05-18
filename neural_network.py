import numpy as np

class Affine:
    def __init__(self, input_size, output_size):
        # Xavierの初期値
        self.weight = np.random.randn(input_size, output_size) * (1 / output_size)**0.5
        self.prev = None 

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight)
    
    def backward(self, dy):
        self.grad = np.multiply(*np.meshgrid(dy, self.x))
        dy = np.dot(dy, self.weight.T)
        return dy

    def step(self, lr, alpha):
        if self.prev is None:
            self.weight += lr*self.grad
            self.prev = lr*self.grad
        else:
            self.weight += lr*self.grad + alpha*self.prev
            self.prev = lr*self.grad + alpha*self.prev