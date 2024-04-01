
# Task 1 E & F: Read the network and run it on the test set and test the network on new inputs

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image,  ImageOps, ImageEnhance


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
    return model

def plot_images(images, labels, preds):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        ax.set_title(f'Pred: {preds[i]}, True: {labels[i]}')
        ax.axis('off')
    plt.show()

def process_custom_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0) 
    img_array = np.array(img) / 255.0
    img_array = 1.0 - img_array
    img_array = (img_array - 0.1307) / 0.3081
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).float()
    return img_tensor

def classify_and_plot(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    plt.imshow(image_tensor.squeeze().numpy(), cmap='binary')
    plt.title(f'Predicted: {pred.item()}')
    plt.axis('off')
    plt.show()


def process_and_classify_custom_images(model, image_paths):
    for image_path in image_paths:
        img_tensor = process_custom_image(image_path)
        classify_and_plot(model, img_tensor)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    model = load_model('./model/model.pth')
    print(model)
# Test the trained model on the test set in MNIST dataset and visualize the first 9 digits
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].squeeze().numpy()
            correct = target.numpy()
            output_values = np.round(output.numpy(), 2)
            
            for i in range(10):
                print(f'Output Values: {output_values[i]} Max Index: {pred[i]} Correct Label: {correct[i]}')
            
            plot_images(data.numpy(), correct, pred)
            break  
# Test the trained model on new inputs
    custom_image_paths = ['./task1_self_test_data/0.png', './task1_self_test_data/1.png', './task1_self_test_data/2.png',
                          './task1_self_test_data/3.png', './task1_self_test_data/4.png', './task1_self_test_data/5.png',
                          './task1_self_test_data/6.png', './task1_self_test_data/7.png', './task1_self_test_data/8.png',
                          './task1_self_test_data/9.png']
    process_and_classify_custom_images(model, custom_image_paths)

if __name__ == '__main__':
    main()
