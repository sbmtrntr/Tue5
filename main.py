import numpy as np
import function as F
import neural_network as nn
import data_loader as dl

# 再現性を保つためにseed値を固定
seed = 0
np.random.seed(seed)

INPUT_NUM = 64  # 入力の数
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

# 損失関数 : 交差エントロピーを使用
criterion = F.cross_entropy_error

# train_data0 : 1人目の学習用データ
# train_data1 : 2人目の学習用データ
# validation_data0 : 1人目のテスト用データ
# validation_data1 : 2人目のテスト用データ
# すべて　20(あ～と) x 100(文字) x 64(メッシュ特徴量)　で構成
train_data0 = dl.get_data('0', 'L')
train_data1 = dl.get_data('1', 'L')
validation_data0 = dl.get_data('0', 'T')
validation_data1 = dl.get_data('1', 'T')

# label : 正解ラベルのデータ
