import matplotlib.pyplot as plt
import pandas as pd

def make_graph():
    df = [pd.read_csv(f'Results/Task{i}.csv') for i in range(1, 4)]
    fig, ax = plt.subplots(3, 2)
    for i in range(3):
        ax[i, 0].plot(df[i]['Epochs'], df[i]['Accuracy'], color="blue")
        ax[i, 0].set_xlabel('Epochs')
        ax[i, 0].set_ylabel('Accuracy')
        ax[i, 0].set_title(f'Task{i+1}')

        ax[i, 1].plot(df[i]['Epochs'], df[i]['Loss'], color="red")
        ax[i, 1].set_xlabel('Epochs')
        ax[i, 1].set_ylabel('Loss')
        ax[i, 1].set_title(f'Task{i+1}')
    plt.tight_layout()
    plt.savefig("Results/Results.png")