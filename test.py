import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable



    
    
data_transforms = transforms.Compose([transforms.Resize([512, 512]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5,], [0.5,])
                                     ])
    
data_dir = 'data/Images' 
image_datasets = datasets.ImageFolder(data_dir, transform = data_transforms)
train_loader = torch.utils.data.DataLoader(dataset = image_datasets, batch_size =2, shuffle = True)

image_datasets.classes
labels_h = ('Circle', 'Rectangle', 'Triangle')
dataset_size = len(image_datasets)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        #convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16,
                              kernel_size=5, stride = 1 , padding = 2)
        self.relu1 = nn.ReLU()
        #MaxPool1
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        #convolution2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=5, stride = 1 , padding = 2)
        self.relu2 = nn.ReLU()
        #MaxPool2
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.fc1 = nn.Linear(2097152, 7)
        
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        
        out = self.maxpool1(out)
        
        out = self.cnn2(out)
        out = self.relu2(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        return out
    
    
model = CNNModel()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr = learning_rate)


#Training the model

epochs = 5
iter = 0
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loader):
        #Loading the images in Variable
        images = Variable(images)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        output = model(images)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        iter = iter + 1
        print('Epoch [%d/%d], iter [%d/%d] loss : %.4f' % (epoch+1 , epochs, i+1, len(image_datasets) // dataset_size, loss.data))
        
        
        
data_dir ='data/Images/train'
test_image_datasets = datasets.ImageFolder(data_dir, transform = data_transforms)
test_loader = torch.utils.data.DataLoader(dataset = test_image_datasets, batch_size = 1, shuffle = True)


dataset_size1 = len(test_image_datasets)


model.eval()
correct =0
total = 0

for i, (images,labels) in enumerate(test_loader):
    images = torch.randn(1, 3, 512, 512)
    plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
    images = Variable(images)
    labels = Variable(labels)
    
    outputs =model(images)
    _, predicted =torch.max(outputs.data, 1)
    
    print("Prediction -- ", labels_h[predicted])
print(images.shape)  