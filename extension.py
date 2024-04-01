
# Analyze and visualize the 1st layer of Resnet18
import torch
import torchvision.models as models
import torchvision.models.resnet as resnet
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)

conv1_weights = model.conv1.weight.data

filters = conv1_weights.cpu().numpy()

fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[0]:
        ax.imshow(filters[i, 0], cmap='gray')  
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()

image_path = './greek_train/alpha/alpha_001.png'  

if not os.path.exists(image_path):
    raise FileNotFoundError(f"The specified image file does not exist: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Failed to load image from {image_path}")

image = cv2.resize(image, (224, 224))  

filtered_images = []
with torch.no_grad():
    for i in range(filters.shape[0]):
        kernel = filters[i, 0]
        filtered_image = cv2.filter2D(image, -1, kernel)
        filtered_images.append(filtered_image)

fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    if i < len(filtered_images):
        ax.imshow(filtered_images[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
