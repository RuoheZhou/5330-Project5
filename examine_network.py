import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import cv2  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def apply_filters_and_plot(image, weights):
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))  
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            filter = weights[i, 0].numpy()
            filtered_image = cv2.filter2D(image.numpy(), -1, filter)
            ax.imshow(filtered_image, cmap='gray')  
            ax.set_title(f'Filter {i}')
            ax.axis('off')
    plt.savefig('./filter_plot2.png')
    plt.show()

def read_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def filter_plot(weights):
    
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:  
            ax.imshow(weights[i, 0], cmap='Pastel2')
            ax.set_title(f'Filter {i}')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off') 
    plt.savefig('./filter_plot1.png')
    plt.show()

def main():
    model_path = './model/model.pth'  
    model = read_model(model_path)
    print(model)
    weights = model.conv1.weight.data
    print("weights' shape:", weights.shape)
    print("filter weights of 1st layer:", weights[0, 0])
    
    filter_plot(weights)
    plt.pause(1)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    first_image, _ = train_dataset[0]
    weights = model.conv1.weight.detach()
    apply_filters_and_plot(first_image[0], weights) 
    plt.pause(1)

if __name__ == '__main__':
    main()
