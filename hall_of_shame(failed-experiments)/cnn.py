
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                test=False,
                shuffle=True):

    # Keeps the normalization and 227x227 resize from the blog
    normalize = transforms.Normalize(
                          mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010])

    transform = transforms.Compose([
                          transforms.Resize((227,227)),
                          transforms.ToTensor(),
                          normalize ])

    # 1. Handle Test Mode
    if test:
        # Points to data_dir/testing
        test_dir = os.path.join(data_dir, '/testing')
        dataset = datasets.ImageFolder(root=test_dir, transform=transform)

        data_loader = torch.utils.data.DataLoader(
                            dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader

    # 2. Handle Train/Validation Mode
    # Points to data_dir/train
    train_dir = os.path.join(data_dir, 'train')
    
    # Create two instances pointing to the same training folder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


data_dir = "C:/Users/Row-Huh/Documents/projecs/naruto-recognition/kaggle/naruto-hand-signs/data"

train_loader, valid_loader = data_loader(data_dir=data_dir, batch_size=64)
test_loader = data_loader(data_dir=data_dir, batch_size=64, test=True)


# VGG-16 model. Each nn.sequential block represents a sequence of layers. Layers 1 to 13: 
# It will be Convolutional layers with batch normalization 
# and ReLU activation functions. Max pooling is applied after layers 2, 4, 7, and 10 to reduce spatial dimensions.

class CNN(nn.Module):
    def __init__(self, )

num_classes = 100
num_epochs = 20
batch_size = 16
learning_rate = 0.005

model = VGG16(num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  


total_step = len(train_loader)
# training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Take the Tensors onto the device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))