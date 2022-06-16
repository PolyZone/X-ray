import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def loss_plots(train_loss, valid_loss):
    plt.plot(train_loss, label="Training loss")
    plt.plot(valid_loss, label="Validation loss")
    plt.legend()

    plt.title('Training and validation loss \n With dropout-layers and lr=0.0001 og L2=0.01')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    return plt.show()

def acc_plots(train_acc, valid_acc):
    plt.plot(train_acc, label="Training accuracy")
    plt.plot(valid_acc, label="Validation accuracy")
    plt.legend()

    plt.title('Training and validation accuracy \n With dropout-layers and lr=0.0001 og L2=0.01')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    return plt.show()


def plot_confusion_matrix(y_label, y_pred, title = "Confusion Matrix"):
    cm = confusion_matrix(y_label, y_pred)

    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax,linewidths=0.1, annot_kws={"size":15});  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize = 15);ax.set_ylabel('True labels', fontsize = 15);
    ax.set_title(title, fontsize = 20);
    ax.xaxis.set_ticklabels(['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST'], fontsize = 12); ax.yaxis.set_ticklabels(['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST'], fontsize = 12);
    return plt.show()


