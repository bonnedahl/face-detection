from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import numpy as np

# check GPU availability
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
#
#vgg16 = models.vgg16(pretrained=True)
# vgg16.to(device)
# print(vgg16)

size = 224
dim = (size, size)

# funktion för att konvertera 28x28, gråskaligabilder till 32x32 i färg

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


traindir = './train_imagestest'
testdir = "./test_imagestest"


# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    # Validation does not use augmentation
    'test':
    transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}

batch_size = 32

# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
print(features.shape, labels.shape)

model = models.vgg16(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Add on classifier
n_classes = 2
print("Before")
print(model)

# model.classifier[6] = nn.Sequential(
#   nn.Linear(n_inputs, 256),
#  nn.ReLU(),
# nn.Dropout(0.4),
#nn.Linear(256, n_classes),
# nn.LogSoftmax(dim=1))

num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
# Add our layer with 4 outputs
features.extend([nn.Linear(num_features, 2)])
model.classifier = nn.Sequential(*features)  # Replace the model classifier
print("After")
print(model)
