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

net = model
