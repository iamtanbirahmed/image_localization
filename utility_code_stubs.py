import torch

import torchvision
from PIL import Image
from matplotlib import patches
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class box_transform():
    def __init__(self, dimension, size):
        self.dimention = dimension
        self.size = size

    def transform(self):
        self.dimention = self.dimention.split(' ')
        self.dimention = list(map(float, self.dimention))
        self.dimention = np.array(self.dimention, dtype='float32')

        self.dimention[0] = float(self.dimention[0]) / self.size[0] * 224
        self.dimention[1] = float(self.dimention[1]) / self.size[1] * 224
        self.dimention[2] = float(self.dimention[2]) / self.size[0] * 224
        self.dimention[3] = float(self.dimention[3]) / self.size[1] * 224

        self.dimention = torch.tensor(self.dimention)
        return self.dimention


class ILOTestDataset(Dataset):
    def __init__(self, transform=None):
        with open('data/test_images.txt') as f:
            self.test_image_paths = [l for l in f.read().splitlines()]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        im = Image.open('./data/images/' + self.test_image_paths[index]).convert('RGB')
        image = self.transform(im)
        return image

    def __len__(self):
        return len(self.test_image_paths)

def test_single_image():
    # 71.7014, 30.3061, 98.5784, 120.8087
    fig, ax = plt.subplots(1)

    im = Image.open('./data/images/' + '001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.jpg').convert('RGB')

    ts = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    ax.imshow(np.transpose(ts(im), (1, 2, 0)))

    # Create a Rectangle patch
    rect = patches.Rectangle((71.7014, 30.3061), 98.5784, 120.8087, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()



def show_images(images):
    image_grid = torchvision.utils.make_grid(images)
    # grid = torchvision.utils.make_grid(images, nrow=10)
    image_grid = image_grid.numpy()
    plt.imshow(np.transpose(image_grid, (1, 2, 0)))
    plt.show()

#
# def plot_test_Image():
#     im = Image.open('./data/images/' + '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg').convert('RGB');
#     fig, ax = plt.subplots(1)
#     ax.imshow(im)
#     rect = patches.Rectangle((60.0, 27.0), 325.0, 304.0, linewidth=1, edgecolor='r', facecolor='none')
#
#     # Add the patch to the Axes
#     ax.add_patch(rect)
#
#     plt.show()
#
#
# def plot_images(images, image_sizes, boxes):
#     fig, ax = plt.subplots(1)
#     dimentions = transform_box_location(boxes[1].split(' '), tensor_to_nd_array(image_sizes[1]))
#     # Display the image
#
#     x, y, w, h = dimentions
#
#     ax.imshow(np.transpose(images[1], (1, 2, 0)))
#
#     # Create a Rectangle patch
#     rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
#
#     # Add the patch to the Axes
#     ax.add_patch(rect)
#
#     plt.show()
