import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        #重み・バイアスの初期値は標準正規分布に従う乱数値
        self.weight = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def __call__(self, input):
        return np.dot(input, self.weight) + self.bias
