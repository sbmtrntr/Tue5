import numpy as np
import functional as F
import neural_network as nn
import data_loader as dl
import make_graph as mg

# 再現性を保つためにseedを固定
seed = 0
np.random.seed(seed)

# ハイパーパラメータ
INPUT_SIZE = 64
HIDDEN_SIZE = 32
OUTPUT_SIZE = 20
LEARNING_RATE = 0.01
ALPHA = 0.9

class MultiLayerPerceptron:
    # 3層パーセプトロン
    def __init__(self):
        self.fc1 = nn.Affine(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Affine(HIDDEN_SIZE, OUTPUT_SIZE)
        self.act1 = F.Sigmoid()
        self.act2 = F.Sigmoid()
        self.criterion = F.MSE()
        self.lr = LEARNING_RATE
        self.alpha = ALPHA

    # 順伝播
    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.act1.forward(x)
        x = self.fc2.forward(x)
        x = self.act2.forward(x)
        return x
    
    # 逆伝播
    def backward(self):
        dy = self.criterion.backward()
        dy = self.act2.backward(dy)
        dy = self.fc2.backward(dy)
        dy = self.act1.backward(dy)
        dy = self.fc1.backward(dy)

    # 重みの更新
    def step(self):
        self.fc1.step(self.lr, self.alpha)
        self.fc2.step(self.lr, self.alpha)


# 使用するデータを取得
train0 = dl.get_data('0', 'L') # 1人目の学習用データ
test0 = dl.get_data('0', 'T') # 1人目のテスト用データ
train1 = dl.get_data('1', 'L') # 2人目の学習用データ
test1 = dl.get_data('1', 'T') # 2人目のテスト用データ
y_true = dl.get_label() # 正解ラベルのデータ (one-hot)

#################################################################################################

# ニューラルネットワークのインスタンスを生成
MLP1 = MultiLayerPerceptron()

average_loss = 1.0 # 損失の平均
epochs = 0 # エポック数
accuracy_result = [] # 図の描写用リスト
loss_result = [] # 図の描写用リスト
f = open("Results/Task1.txt", 'w') # 結果保存用のファイルを開いておく

# 1. 筆記者0の学習用データでニューラルネットの学習を行う
while average_loss > 0.001:
    running_loss = 0.0 # 1epochの損失の合計
    accuracy_cnt = 0  # 正答数
    epochs += 1
    for i in range(100):
        for j in range(20):
            y_pred = MLP1.forward(train0[j, i]) # 順伝播
            loss = MLP1.criterion(y_pred, y_true[j, i]) # 損失の計算
            MLP1.backward() # 逆伝播
            MLP1.step() # 重みの更新
            running_loss += loss
            accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[j, i])

    accuracy = accuracy_cnt / 2000 # 正答率
    average_loss = running_loss / 2000 # 損失の平均

    print("Epochs:", str(epochs), "Accuracy:", str(accuracy), "Loss:", str(average_loss))
    f.writelines(["Epochs:", str(epochs), "  Accuracy:", str(accuracy), "  Loss:", str(average_loss), '\n'])
    accuracy_result.append(accuracy)
    loss_result.append(average_loss)

# 図の描写
mg.make_graph("Task1.png", "Training with data from writer0", accuracy_result, loss_result)

# 2. 1で学習したニューラルネットに筆記者0の学習用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP1.forward(train0[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("train0 Accuracy:", str(accuracy_cnt / 2000))
f.writelines(["train0 Accuracy:", str(accuracy_cnt / 2000), '\n'])


# 3. 1で学習したニューラルネットに筆記者0のテスト用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP1.forward(test0[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("test0 Accuracy:", str(accuracy_cnt / 2000))
f.writelines(["test0 Accuracy:", str(accuracy_cnt / 2000), '\n'])


# 4. 1で学習したニューラルネットに筆記者1のテスト用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP1.forward(test1[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("test1 Accuracy:", str(accuracy_cnt / 2000))
f.writelines(["test1 Accuracy:", str(accuracy_cnt / 2000), '\n'])
f.close()

#################################################################################################

# 新しくニューラルネットワークのインスタンスを生成
MLP2 = MultiLayerPerceptron()

average_loss = 1.0 # 損失の平均
epochs = 0 # エポック数
accuracy_result = []
loss_result = []
f = open("Results/Task2.txt", 'w')

# 5. 筆記者1の学習用データでニューラルネットの学習を行う
while average_loss > 0.001:
    running_loss = 0.0 # 1epochの損失の合計
    accuracy_cnt = 0  # 正答数
    epochs += 1
    for i in range(100):
        for j in range(20):
            y_pred = MLP2.forward(train1[j, i]) # 順伝播
            loss = MLP2.criterion(y_pred, y_true[j, i]) # 損失の計算
            MLP2.backward() # 逆伝播
            MLP2.step() # 重みの更新
            running_loss += loss
            accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[j, i])

    accuracy = accuracy_cnt / 2000 # 正答率
    average_loss = running_loss / 2000 # 損失の平均

    print("Epochs:", str(epochs), "Accuracy:", str(accuracy), "Loss:", str(average_loss))
    f.writelines(["Epochs:", str(epochs), "  Accuracy:", str(accuracy), "  Loss:", str(average_loss), '\n'])
    accuracy_result.append(accuracy)
    loss_result.append(average_loss)

# 図の描写
mg.make_graph("Task2.png", "Training with data from writer1", accuracy_result, loss_result)

# 6. 5で学習したニューラルネットに筆記者1の学習用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP2.forward(train1[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("train0 Accuracy:", str(accuracy_cnt / 2000))
f.writelines(["train0 Accuracy:", str(accuracy_cnt / 2000), '\n'])


# 7. 5で学習したニューラルネットに筆記者0のテスト用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP2.forward(test0[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("test0 Accuracy:", str(accuracy_cnt / 2000))
f.writelines(["test0 Accuracy:", str(accuracy_cnt / 2000), '\n'])


# 8. 5で学習したニューラルネットに筆記者1のテスト用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP2.forward(test1[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("test1 Accuracy:", str(accuracy_cnt / 2000))
f.writelines(["test1 Accuracy:", str(accuracy_cnt / 2000), '\n'])
f.close()

#################################################################################################

# 新しくニューラルネットワークのインスタンスを生成
MLP3 = MultiLayerPerceptron()

average_loss = 1.0 # 損失の平均
epochs = 0 # エポック数
accuracy_result = []
loss_result = []
f = open("Results/Task3.txt", 'w')

# 9. 筆記者0と筆記者1の学習用データでニューラルネットの学習を行う
while average_loss > 0.001:
    running_loss = 0.0 # 1epochの損失の合計
    accuracy_cnt = 0  # 正答数
    epochs += 1
    for i in range(100):
        for j in range(20):
            y_pred = MLP3.forward(train0[j, i]) # 順伝播
            loss = MLP3.criterion(y_pred, y_true[j, i]) # 損失の計算
            MLP3.backward() # 逆伝播
            MLP3.step() # 重みの更新
            running_loss += loss
            accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[j, i])

            y_pred = MLP3.forward(train1[j, i]) # 順伝播
            loss = MLP3.criterion(y_pred, y_true[j, i]) # 損失の計算
            MLP3.backward() # 逆伝播
            MLP3.step() # 重みの更新
            running_loss += loss
            accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[j, i])

    accuracy = accuracy_cnt / 4000 # 正答率
    average_loss = running_loss / 4000 # 損失の平均

    print("Epochs:", str(epochs), "Accuracy:", str(accuracy), "Loss:", str(average_loss))
    f.writelines(["Epochs:", str(epochs), "  Accuracy:", str(accuracy), "  Loss:", str(average_loss), '\n'])
    accuracy_result.append(accuracy)
    loss_result.append(average_loss)


# 図の描写
mg.make_graph("Task3.png", "Training with data from writer0 and writer1", accuracy_result, loss_result)

# 10. 9で学習したニューラルネットに筆記者0と筆記者1の学習用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP3.forward(train0[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])
        y_pred = MLP3.forward(train1[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("train0 and train1 Accuracy:", str(accuracy_cnt / 4000))
f.writelines(["train0 and train1 Accuracy:", str(accuracy_cnt / 4000), '\n'])


# 11. 9で学習したニューラルネットに筆記者0と筆記者1のテスト用データを入力して識別を行う
accuracy_cnt = 0
for i in range(20):
    for j in range(100):
        y_pred = MLP3.forward(test0[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])
        y_pred = MLP3.forward(test1[i, j])
        accuracy_cnt += np.argmax(y_pred) == np.argmax(y_true[i, j])

print("test0 and test1 Accuracy:", str(accuracy_cnt / 4000))
f.writelines(["test0 and test1 Accuracy:", str(accuracy_cnt / 4000), '\n'])
f.close()