import pretrainedmodels
import cnn_finetune
import os
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil
import random


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def correction(data_dir):
    temp = os.path.split(data_dir)
    root = temp[0]
    new_path = os.path.join(root, "Balanced")
    if (os.path.isdir(new_path) == False):
        os.mkdir(new_path)
    relative = os.listdir(data_dir)
    # print(relative)
# For NOFINDING
    pre_nofind = os.path.join(data_dir, "NOFINDING")
    current_nofind = os.path.join(new_path, "NOFINDING")
    if (os.path.isdir(current_nofind) == False):
        os.mkdir(current_nofind)
    NOFIND = os.listdir(pre_nofind)
    # print(NOFIND[0])
    ran = list(random.sample(NOFIND, 363))
    for r in ran:
        old = os.path.join(pre_nofind, r)
        new = os.path.join(current_nofind, r)
        shutil.copy2(old, new)
    if (os.path.isdir(pre_nofind) == True):
        shutil.rmtree(pre_nofind)
# For Covid
    previous = os.path.join(data_dir, "COVID-19")
    print(previous)
    current_COVID = os.path.join(new_path, "COVID-19")
    if (os.path.isdir(current_COVID) == False):
        os.mkdir(current_COVID)
    copytree(previous, current_COVID)
    if (os.path.isdir(previous) == True):
        shutil.rmtree(previous)
# For The Thoraxdisease
    pre_thor = os.path.join(data_dir, "THORAXDISEASE")
    current_thor = os.path.join(new_path, "THORAXDISEASE")
    if (os.path.isdir(current_thor) == False):
        os.mkdir(current_thor)
    THOR = os.listdir(pre_thor)
    # print(NOFIND[0])
    ran = list(random.sample(THOR, 363))
    for r in ran:
        old = os.path.join(pre_thor, r)
        new = os.path.join(current_thor, r)
        shutil.copy2(old, new)
    if (os.path.isdir(pre_thor) == True):
        shutil.rmtree(pre_thor)
    shutil.rmtree(data_dir)
    return new_path


# Make a balanced dataset
data_dir = 'CSC413Project/COVID-19 Radiography Database'
data_dir = correction(data_dir)


# 0:Covid 1: NoFinding 2: Thorax
def dataload(batch_size=20):
    # define training and test data directories

    # define dataloader parameters
    num_workers = 0

    # Validation and Testing - just resize and crop the images
    data_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(data_dir, transform=data_transform)

    dataset = torch.utils.data.Subset(
        dataset, np.random.choice(len(dataset), 512, replace=False))

    print("Total number of images loaded:", len(dataset))
    train, valid, test = torch.utils.data.random_split(dataset, [round(
        0.7*len(dataset)), round(0.2*len(dataset)), len(dataset)-round(0.7*len(dataset))-round(0.2*len(dataset))])
    trainLoader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)
    validLoader = torch.utils.data.DataLoader(valid, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)
    testLoader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=True)

    return trainLoader, validLoader, testLoader


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight
