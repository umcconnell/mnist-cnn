#!/usr/bin/env python
# coding: utf-8

# # mnist-cnn

# Let's build a super basic Convolutional Neural Network(CNN) to classify MNIST handwritten digits! We'll be using pytorch to specify and train our network.

# ## Setup

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


# In[ ]:


EPOCHS = 2
BATCH_SIZE = 64

NUM_CLASSES = 10


# ## Loading MNIST Dataset
# 
# We'll be using the MNIST dataset to train our CNN. It contains images of handwritten digits. Loading MNIST is trivial using torchvision.
# 
# Before we can use the images to train the network, it's a best practice to normalize the images. The images are black-and-white, represented by values from [0, 1]. The transformation will bring the values in a range of [-1, 1]:

# In[ ]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


# In[ ]:


trainset = torchvision.datasets.MNIST(
    root="./data", download=True, train=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data", download=True, train=False, transform=transform
)


# In[ ]:


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


# ## Visualizing
# 
# Let's visualize the dataset before actually using it:

# In[ ]:


def show_img(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(npimg[:, :], cmap='gray_r')
    plt.show()


# In[ ]:


dataiter = iter(trainloader)
imgs, labels = next(dataiter)
show_img(imgs[0].squeeze())
print('Label: %i' % labels[0].item())


# ## Model
# 
# Now we can at last define our CNN. It consists of:
# - two convolutional blocks to extract relevant features from the input image
# - three fully connected layers to process the extracted features and classify the digit images

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(250, 120),
            nn.Linear(120, 60),
            nn.Linear(60, NUM_CLASSES)
        )

    def forward(self, x):
        return self.model(x)


# ## Training
# 
# First we'll try setting up pytorch to use a CUDA-capable GPU. If no GPU is detected, our CNN will be trained on CPU:

# In[ ]:


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device "%s" for training' % dev)


# We then create an instance of our network before moving it to our training device using the `.to()` method:

# In[ ]:


neural_net = Net().to(dev)


# Next, we'll define our loss and optimizer:

# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(neural_net.parameters())


# Now, let's train our network!

# In[ ]:


for epoch in range(EPOCHS):
    running_loss = 0.0
    
    for i, (imgs, labels) in enumerate(trainloader):
        imgs = imgs.to(dev)
        labels = labels.to(dev)

        # Important!
        # Clear accumulated gradients from previous iteration
        # before backpropagating. 
        optimizer.zero_grad()
        
        y = neural_net(imgs)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%.3d / %.3d] Loss: %.9f' % (epoch, i, running_loss / 100))


# ## Testing
# 
# Finally, let's test the performance of our network on the testset, containing images of digits the network hasn't seen before:

# In[ ]:


neural_net.train(False)
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        images, labels = data
        images = images.to(dev)
        labels = labels.to(dev)
        
        outputs = neural_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

