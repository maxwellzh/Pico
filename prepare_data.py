import torch
import numpy as np
from PIL import Image
import os
from torchvision import datasets, transforms

datasets.MNIST('./data', train=True, download=True)

source_dir = 'data/MNIST/processed/'
target_dir = 'data/MNIST'

training_images = torch.load(source_dir + '/training.pt')
test_images = torch.load(source_dir + '/test.pt')

# train set
traindir = target_dir + '/train/'
os.makedirs(traindir, exist_ok=True)
images = training_images[0].numpy()
labels = training_images[1].numpy()
print("Images shape:", images.shape)
print("Label shape:", labels.shape)
size = labels.shape[0]
for i in range(size):
    l = labels[i].item()
    os.makedirs(traindir + str(l), exist_ok=True)
    img = Image.fromarray(images[i])
    img.save(traindir + '{}/{}_{:05}.bmp'.format(l, l, i))
    print('\rProcessed: [{}/60000]'.format(i+1), end='')
print('')

# test set
testdir = target_dir + '/test/'
os.makedirs(traindir, exist_ok=True)
images = test_images[0].numpy()
labels = test_images[1].numpy()

size = labels.shape[0]
for i in range(size):
    l = labels[i].item()
    os.makedirs(testdir + str(l), exist_ok=True)
    img = Image.fromarray(images[i])
    img.save(testdir + '{}/{}_{:05}.bmp'.format(l, l, i))
    print('\rrProcessed: [{}/10000]'.format(i+1), end='')

print('')
