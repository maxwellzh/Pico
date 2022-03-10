
import os
# tested with
# torch==1.10.2
# torchvision==0.11.3
from torchvision import datasets

target_dir = 'data/MNIST'
os.makedirs(target_dir, exist_ok=True)

# images will be stored at data/train and data/test
for subset in ['train', 'test']:
    print(f"Processing {subset} set:")
    train_images = datasets.MNIST(
        './data', train=(subset == 'train'), download=True)
    traindir = os.path.join(target_dir, subset)
    os.makedirs(traindir, exist_ok=True)
    size = len(train_images)
    for i in range(size):
        img, label = train_images[i]
        os.makedirs(os.path.join(traindir, str(label)), exist_ok=True)
        img.save(os.path.join(traindir, '{0}/{0}_{1:05}.bmp'.format(label, i)))
        print('\rProcessing: [{}/{}]'.format(i+1, size), end='')
    print('')
