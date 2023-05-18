import matplotlib.pyplot as plt

def make_graph(name, title, accuracy_result, loss_result):
    fig = plt.figure()
    ax1 = fig.subplots()
    ax1.set_title(title)
    ax1.set_xlabel("Epochs")
    ax2 = ax1.twinx()
    ax1.plot(range(len(accuracy_result)), accuracy_result, color="blue", label="Accuracy")
    ax1.set_ylabel("Accuracy")
    ax2.plot(range(len(loss_result)), loss_result, color="red", label="Loss")
    ax2.set_ylabel("Loss")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc='upper left')
    plt.savefig("Results/" + name)
