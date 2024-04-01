# task 2 part 3: Transfer Learning on Greek Letters
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
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
def preprocess_and_load_images(image_paths, transform):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image.unsqueeze(0))  
    return torch.cat(images, dim=0)  

def classify_images(model, images, class_names):
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for idx, pred in enumerate(predicted):
            print(f'Image {idx + 1}: Predicted class is {class_names[pred.item()]}')

class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

def preprocess_and_load_images(image_paths):

    transform_pipeline = transforms.Compose([
        GreekTransform(),
        transforms.ToTensor(),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
       
        image = transform_pipeline(image)
        images.append(image.unsqueeze(0))  
    return torch.cat(images, dim=0)  

def process_custom_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
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
        shuffle=False
    )
    
    model = load_model('./model/model.pth')
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    training_errors = []
    epochs = 800
    perfect_threshold = 0.99  
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for data, target in greek_train:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(greek_train.dataset)
        accuracy = correct / len(greek_train.dataset)

        training_errors.append(avg_loss)
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        if accuracy >= perfect_threshold:
            print(f"Reached almost perfect accuracy after {epoch + 1} epochs.")
            break
    
    image_paths = [
        './handwritten_data/test1.png', './handwritten_data/test2.png', './handwritten_data/test3.png',
        './handwritten_data/test4.png', './handwritten_data/test5.png', './handwritten_data/test6.png',
        './handwritten_data/test7.png', './handwritten_data/test8.png', './handwritten_data/test9.png'
    ]
    process_and_classify_custom_images(model, image_paths, i)
    
if __name__ == '__main__':
    main()

