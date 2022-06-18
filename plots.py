import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
#import McNemar from class



def loss_plots(train_loss, valid_loss,title="Training and validation loss"):
    plt.plot(train_loss, label="Training loss")
    plt.plot(valid_loss, label="Validation loss")
    plt.legend()

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    return plt.show()

def acc_plots(train_acc, valid_acc, title="Training and validation accuracy"):
    plt.plot(train_acc, label="Training accuracy")
    plt.plot(valid_acc, label="Validation accuracy")
    plt.legend()

    plt.title(title)
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
# https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels

def McNemar_plot(all_labels, models_names_pred, total_models, title="McNemar Test"):
    import seaborn as sns
    from matplotlib.patches import Rectangle
    matrix = np.zeros((total_models, total_models))
    matrix_tom = np.zeros((total_models, total_models), str)

    model_name = models_names_pred
    all_labels = torch.load(all_labels)
    matrix = np.zeros((total_models, total_models))
    for i in range(total_models):

        pred_1 = torch.load(model_name[i])
        for n in range(total_models):
            pred_2 = torch.load(model_name[n])
            p_val = testMcNemar(pred_1, pred_2, all_labels)
            matrix[i][n] = p_val
            if matrix[i][n] < 0.05:
                matrix_tom[i][n] = "*"


    labels = (np.asarray(["{0} {1:.5f}".format(matrix_tom, value)
                          for matrix_tom, value in zip(matrix_tom.flatten(),
                                                       matrix.flatten())])
              ).reshape(total_models, total_models)

    fig, ax = plt.subplots(figsize=(15, 10))


    ticks_lab = ["Model " + str((i+1)) for i in (range(total_models))]

    ax.set_title(title, fontsize=20);
    sns.heatmap(matrix, annot=labels, square=True, vmin=0, vmax=0, linecolor='k', cmap="Blues_r", fmt="", ax=ax,
                linewidth=0.5,xticklabels=ticks_lab, yticklabels=ticks_lab)

    return plt.show()
    #https://stackoverflow.com/questions/41164710/how-to-add-text-plus-value-in-python-seaborn-heatmap


# models_names_pred = [r"model%s_final_predictions.pt" % i for i in range(69, 79)]
# McNemar_plot("all_labels", models_names_pred, total_models=len(models_names_pred), title="McNemar Test Baseline")