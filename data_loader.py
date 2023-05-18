import numpy as np

# x : 0 or 1
# y : L or T
def get_data(x, y):
    # 20(あ～と) x 100(文字) x 64(メッシュ特徴量)
    mesh_data = np.zeros((20, 100, 64))
    for i in range(20):
        if i <= 9:
            text = np.loadtxt('Data/hira' + x + '_0'+ str(i) + y + '.dat', dtype='str')
        else:
            text = np.loadtxt('Data/hira' + x + '_' + str(i) + y + '.dat', dtype='str')
    
        temp = np.array([list(map(float, list(line.rstrip()))) for line in text])

        for j in range(100):
            for col in range(8):
                for row in range(8):
                    block = temp[64*j + col*8 : 64*j + col*8 + 8, row*8: row*8 + 8]
                    mesh_data[i, j, 8*col + row] = np.sum(block) / 64
    
    return mesh_data

def get_label():
    label = np.ndarray((20, 100, 20))
    for i in range(20):
        for j in range(100):
            for k in range(20):
                if i == k:
                    label[i, j, k] = 1
                else:
                    label[i, j, k] = 0
    return label