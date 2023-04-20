import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocess import correction, dataload


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # normalize the confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print("confusion matrix:", cm)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(dataLoader):
    model = torch.load('/CSC413Project/ModelCheckPoint/0.9705882352941176')

    use_cuda = True
    test_labels = []
    pred_labels = []
    for imgs, labels in dataLoader:
        ############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        ############################################
        output = model(imgs)
        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1].reshape(1, -1)
        pred_labels.append(pred.tolist()[0])
        test_labels.append(labels.tolist())

    test_labels = [val for sublist in test_labels for val in sublist]
    pred_labels = [val for sublist in pred_labels for val in sublist]
    print(pred_labels)
    print(test_labels)
    print('Confusion Matrix')
    cm = confusion_matrix(test_labels, pred_labels)
    plot_confusion_matrix(
        cm, classes=['COVID-19', 'NOFINDING', 'THORAXDISEASE'], title='Confusion Matrix')
    print('Classification Report')
    print(classification_report(test_labels, pred_labels,
          target_names=['NOFINDING', 'THORAXDISEASE', 'COVID-19']))


if __name__ == '__main__':
    # Make a balanced dataset
    data_dir = 'CSC413Project/COVID-19 Radiography Database'
    data_dir = correction(data_dir)
    # Load the data
    trainLoader, validLoader, testLoader = dataload(data_dir=data_dir)

    # Test the model
    test(validLoader)

    test(testLoader)
