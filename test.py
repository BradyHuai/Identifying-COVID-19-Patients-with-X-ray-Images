import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import cv2
import os
from PIL import Image
from operator import add
from functools import reduce
from google.colab import drive
from sklearn.metrics import classification_report, confusion_matrix
import pretrainedmodels
import cnn_finetune
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
          plt.text(j, i, cm[i, j],
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    model = torch.load('/CSC413Project/ModelCheckPoint/0.9705882352941176')

    use_cuda = True
    test_labels = []
    pred_labels = []
    for imgs, labels in validLoader:
        ############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
        imgs = imgs.cuda()
        labels = labels.cuda()
        ############################################
        output = model(imgs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1].reshape(1,-1)
        pred_labels.append(pred.tolist()[0])
        test_labels.append(labels.tolist())
    #print(pred_labels)
    test_labels = [val for sublist in test_labels for val in sublist]
    pred_labels = [val for sublist in pred_labels for val in sublist]
    print(pred_labels)
    print(test_labels)
    print('Confusion Matrix')
    cm = confusion_matrix(test_labels,pred_labels)
    plot_confusion_matrix(cm, classes=['COVID-19','NOFINDING','THORAXDISEASE'],title='Confusion Matrix')
    print('Classification Report')
    print(classification_report(test_labels,pred_labels,target_names=['NOFINDING','THORAXDISEASE','COVID-19']))


def test():
    model = torch.load('/content/drive/MyDrive/APS360/Project/ModelCheckPoint/0.9705882352941176')

    use_cuda = True
    test_labels = []
    pred_labels = []
    for imgs, labels in testLoader:
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
        imgs = imgs.cuda()
        labels = labels.cuda()
        #############################################
        output = model(imgs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1].reshape(1,-1)
        pred_labels.append(pred.tolist()[0])
        test_labels.append(labels.tolist())
    #print(pred_labels)
    test_labels = [val for sublist in test_labels for val in sublist]
    pred_labels = [val for sublist in pred_labels for val in sublist]
    print(pred_labels)
    print(test_labels)
    print('Confusion Matrix')
    cm = confusion_matrix(test_labels,pred_labels)
    plot_confusion_matrix(cm, classes=['COVID-19','NOFINDING','THORAXDISEASE'],title='Confusion Matrix')
    print('Classification Report')
    print(classification_report(test_labels,pred_labels,target_names=['NOFINDING','THORAXDISEASE','COVID-19']))

if __name__ == '__main__':
    main()

    test()
