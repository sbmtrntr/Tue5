import numpy as np

def get_data(x, y):
    data = np.zeros((20, 100, 64))
    for i in range(20):
        if i <= 9:
            text = np.loadtxt('Data\hira' + x + '_0'+ str(i) + y + '.dat', dtype='str')
        else:
            text = np.loadtxt('Data\hira' + x + '_' + str(i) + y + '.dat', dtype='str')
        temp_data = [[[] for _ in range(64)] for _ in range(100)]
        for j in range(6400):
            temp_data[j % 100][j % 64].append(list(map(float, list(text[j]))))
        for j in range(100):
            temp = 0.0
            for col in range(64):
                for row in range(64):
                    temp += temp_data[j][col]
            data[i][j][col] = 

    
    return data