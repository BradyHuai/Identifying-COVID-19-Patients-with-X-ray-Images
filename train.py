import pretrainedmodels
import cnn_finetune
import os
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from preprocess import correction, dataload


def make_classifier(in_features, num_classes):
    """
    Creates a linear classifier with dropout and ReLU activation.
    :param in_features: number of input features
    :param num_classes: number of output classes
    :return: nn.Sequential classifier
    """
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes),
    )


def get_accuracy(model, loader):
    """
    Computes the accuracy of the model on the given data.
    :param model: model to evaluate
    :param loader: data loader for the data
    :return: accuracy

    Note:
        - The model is set to evaluation mode before computing the accuracy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)

            # select index with maximum prediction score
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += imgs.size(0)

    model.train()
    return correct / total


def train(model, train_loader, val_loader, batch_size, num_epochs=1, lr=0.01, momentum=0.9):
    """
    Trains the model.

    :param model: model to train
    :param train_loader: data loader for the training data
    :param val_loader: data loader for the validation data
    :param batch_size: batch size
    :param num_epochs: number of epochs
    :param lr: learning rate
    :param momentum: momentum
    :return: None

    Note:
        - The model is set to training mode before training.
        - The model is set to evaluation mode before computing the accuracy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    weights = [1/363, 1/1408, 1/3736]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr)

    iters, losses, train_acc, val_acc = [], [], [], []
    max_val_acc = 0
    min_loss = 10000

    if torch.cuda.is_available():
        print('CUDA is available! Training on GPU ...')
    else:
        print('CUDA is not available. Training on CPU ...')

    # training
    n = 0  # the number of iterations
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        model.train()  # Set model to training mode
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)             # forward pass
            loss = criterion(out, labels)  # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            # compute *average* loss
            losses.append(float(loss)/batch_size)
            n += 1

            del imgs, labels
            torch.cuda.empty_cache()

        model.eval()  # Set model to evaluation mode
        tacc = get_accuracy(model, train_loader)
        vacc = get_accuracy(model, val_loader)
        train_acc.append(tacc)  # compute training accuracy
        val_acc.append(vacc)  # compute validation accuracy

        print('Train acc:', tacc, 'Validation acc:', vacc)
        if loss < min_loss:
            min_loss = loss
            torch.save(model, '/CSC413Project/ModelCheckPoint/'+str(min_loss))
            print('Lowest loss model saved, loss:', min_loss)
        if max_val_acc < vacc:
            max_val_acc = vacc
            torch.save(model, '/CSC413Project/ModelCheckPoint/' +
                       str(max_val_acc))
            print('High val acc model saved, vacc:', max_val_acc)

    # plotting
    plt.title("Training Loss Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Accuracy Curve")
    plt.plot([n for n in range(0, num_epochs)], train_acc, label="Train")
    plt.plot([n for n in range(0, num_epochs)], val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


def main():
    # Make a balanced dataset
    data_dir = 'CSC413Project/COVID-19 Radiography Database'
    data_dir = correction(data_dir)

    # Load the data
    trainLoader, validLoader, testLoader = dataload(data_dir=data_dir)

    batch_size = 32
    learning_rate = 0.001
    epochs = 20
    model = cnn_finetune.make_model('xception', num_classes=3, pretrained=True, input_size=(
        224, 224), classifier_factory=make_classifier)
    train(model, trainLoader, validLoader, batch_size,
          num_epochs=epochs, lr=learning_rate, momentum=0.9)


if __name__ == '__main__':
    main()
