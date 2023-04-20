import os
from torchvision import datasets, transforms
import torch
import numpy as np
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
    root = os.path.split(data_dir)[0]
    new_path = os.path.join(root, "Balanced")
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

    # For NOFINDING
    pre_nofind = os.path.join(data_dir, "NOFINDING")
    current_nofind = os.path.join(new_path, "NOFINDING")
    if not os.path.isdir(current_nofind):
        os.mkdir(current_nofind)
    NOFIND = os.listdir(pre_nofind)

    ran = random.sample(NOFIND, 363)
    for r in ran:
        old = os.path.join(pre_nofind, r)
        new = os.path.join(current_nofind, r)
        shutil.copy2(old, new)
    shutil.rmtree(pre_nofind)

    # For COVID
    previous = os.path.join(data_dir, "COVID-19")
    current_COVID = os.path.join(new_path, "COVID-19")
    if not os.path.isdir(current_COVID):
        os.mkdir(current_COVID)
    for dirpath, dirnames, filenames in os.walk(previous):
        for file in filenames:
            old = os.path.join(dirpath, file)
            new = os.path.join(current_COVID, file)
            shutil.copy2(old, new)
    shutil.rmtree(previous)

    # For THORAXDISEASE
    pre_thor = os.path.join(data_dir, "THORAXDISEASE")
    current_thor = os.path.join(new_path, "THORAXDISEASE")
    if not os.path.isdir(current_thor):
        os.mkdir(current_thor)
    THOR = os.listdir(pre_thor)

    ran = random.sample(THOR, 363)
    for r in ran:
        old = os.path.join(pre_thor, r)
        new = os.path.join(current_thor, r)
        shutil.copy2(old, new)
    shutil.rmtree(pre_thor)

    shutil.rmtree(data_dir)
    return new_path


# targets: 0 - COVID-19, 1 - NOFINDING, 2 - THORAXDISEASE
def dataload(data_dir, batch_size=20):
    # Define dataloader parameters
    num_workers = 0

    # Define data transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset and randomly choose 512 samples
    dataset = datasets.ImageFolder(data_dir, transform=data_transform)
    dataset = torch.utils.data.Subset(
        dataset, np.random.choice(len(dataset), 512, replace=False))

    print("Total number of images loaded:", len(dataset))

    # Split dataset into train, valid, and test sets
    train, valid, test = torch.utils.data.random_split(dataset, [
        round(0.7 * len(dataset)),
        round(0.2 * len(dataset)),
        len(dataset) - round(0.7 * len(dataset)) - round(0.2 * len(dataset))
    ])

    # Create dataloaders for train, valid, and test sets
    trainLoader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    validLoader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    testLoader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return trainLoader, validLoader, testLoader


def make_weights_for_balanced_classes(images, nclasses):
    count = np.zeros(nclasses)
    for item in images:
        count[item[1]] += 1

    weight_per_class = np.zeros(nclasses)
    N = float(len(images))
    weight_per_class = N / count

    weight = [weight_per_class[item[1]] for item in images]

    return weight


def main():
    # Make a balanced dataset
    data_dir = 'CSC413Project/COVID-19 Radiography Database'
    data_dir = correction(data_dir)

    # Load the data
    trainLoader, validLoader, testLoader = dataload(data_dir=data_dir)
