import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

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

def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(model.fc1.out_features, 3)  
    return model

def plot_training_error(training_errors):
    plt.figure(figsize=(10, 5))
    plt.plot(training_errors, label='Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Error Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

def main():
    training_set_path = './greek_train'  
    greek_train = DataLoader(
        torchvision.datasets.ImageFolder(training_set_path, 
                                         transform=transforms.Compose([
                                             GreekTransform(),
                                             transforms.ToTensor(),
                                             transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ])),
        batch_size=9,
        shuffle=True
    )
    
    model = load_model('./model/model.pth')
    print(model)  

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    training_errors = []
    epochs = 800
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in greek_train:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(greek_train)
        
        training_errors.append(avg_loss)
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}')

    plot_training_error(training_errors)

if __name__ == '__main__':
    main()
