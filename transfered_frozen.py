import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torchvision import models

device = "cpu"


data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='./data', transform=data_transforms)
dataset_size = len(dataset)
indices = list(range(dataset_size))
test_split = 0.2
split = int(np.floor(test_split * dataset_size))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)



################################################
# Simply replace the last layer
model = models.alexnet(pretrained = True)

# Set all parameters to not require gradient(will not be optimized)
for parameter in model.parameters():
    parameter.requires_grad = False

# By default, new parameteres have required grad
removed = list(model.classifier.children())[:-1]
model.classifier = nn.Sequential(*removed, nn.Linear(in_features=4096, out_features=3, bias=True))
model = model.to(device)



print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=0.001)

num_epochs = 6
number_of_batches = len(train_loader)

plt.ion()

loss_values = []
final_acc = 0

# Train the network
print("Training set")
total = 0
for epoch in range(num_epochs):

    running_loss = 0.0
    running_corrects = 0    
    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. 
        # These are accumulated into x.grad for every parameter x.


        loss.backward()
        
        # optimizer.step updates the value of x using the gradient x.grad.
        optimizer.step()
        
        

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / (batch_size * len(train_loader))
    epoch_acc = running_corrects.double() / (batch_size * len(train_loader))
    final_acc = epoch_acc
    loss_values.append(epoch_loss)
    plt.plot(loss_values)
    plt.draw()
    plt.pause(0.1)
    print("Epoch: {0}, Loss: {1}, Acc: {2}".format(epoch, epoch_loss, epoch_acc))

print("#" * 40)
print("Test set")
# Test the network
model.eval()
running_loss = 0
running_corrects = 0
total = 0
with torch.no_grad():
    for batch_index, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / (batch_size * len(test_loader))
    epoch_acc = running_corrects.double() / (batch_size * len(test_loader))
    print("Loss: {0}, Acc: {1}".format(epoch_loss, epoch_acc))
