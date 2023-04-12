import numpy as np
import function as F
import NeuralNetwork as nn

#再現性を保つためにseed値を固定
seed = 0
np.random.seed(seed)

INPUT_NUM = 64*64  # 入力の数
HIDDEN_NUM = 256 # 中間層の数
OUTPUT_NUM = 20  # 出力の数

class SimpleNet:
    def __init__(self):
        self.fc1 = nn.Linear(INPUT_NUM, HIDDEN_NUM)
        self.fc2 = nn.Linear(HIDDEN_NUM, HIDDEN_NUM)
        self.fc3 = nn.Linear(HIDDEN_NUM, OUTPUT_NUM)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)
        return x

model = SimpleNet() 

#損失関数 : NLLLossを使用
criterion = F.cross_entropy_error

#train_data : 
train_data = np.loadtxt('Data/hira0_00L.dat', dtype='uint8')
print(train_data)
