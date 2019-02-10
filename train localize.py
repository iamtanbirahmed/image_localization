import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from matplotlib import patches
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np

import progressbar

BATCH_SIZE = 2


##data preprocess

def transform_box_location(dimentions, size):
    dimentions[0] = float(dimentions[0]) / size[0] * 224
    dimentions[1] = float(dimentions[1]) / size[1] * 224
    dimentions[2] = float(dimentions[2]) / size[0] * 224
    dimentions[3] = float(dimentions[3]) / size[1] * 224

    return dimentions


def to_2d_tensor(inp):
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:
        inp = inp.unsqueeze(0)
    return inp


def tensor_to_nd_array(inp):
    inp = inp.numpy()
    return inp[0]


def plot_test_Image():
    im = Image.open('./data/images/' + '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg').convert('RGB');
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    rect = patches.Rectangle((60.0, 27.0), 325.0, 304.0, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


def plot_images(images, image_sizes, boxes):
    fig, ax = plt.subplots(1)
    print(image_sizes[1])
    dimentions = transform_box_location(boxes[1].split(' '), tensor_to_nd_array(image_sizes[1]))
    # Display the image

    x, y, w, h = dimentions

    ax.imshow(np.transpose(images[1], (1, 2, 0)))

    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


def train_network(model):
    EPOCHS = 20
    running_loss = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.SmoothL1Loss()  ## the loss function

    for epoch in range(3):
        print('%d epoch has started' % epoch)
        bar = progressbar.ProgressBar()
        for i, data in enumerate(train_dataloader, 0):
            images, boxes = data
            inputs = images
            targets = boxes
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:  # print every 2000 mini-batches
                print('[epoch %d,batch %d] loss: %.3f' %
                      (epoch, i / 10, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')
    return model


def show_images(images):
    image_grid = torchvision.utils.make_grid(images)
    # grid = torchvision.utils.make_grid(images, nrow=10)
    image_grid = image_grid.numpy()
    plt.imshow(np.transpose(image_grid, (1, 2, 0)))
    plt.show()


def test_single_image():
    # 84.5246, 65.1589, 130.6499, 130.9355
    fig, ax = plt.subplots(1)

    im = Image.open('./data/images/' + '001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.jpg').convert('RGB')

    ts = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    ax.imshow(np.transpose(ts(im),(1,2,0)))

    # Create a Rectangle patch
    rect = patches.Rectangle((84.5246, 65.1589), 130.6499, 130.9355, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


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


class ILODataset(Dataset):
    def __init__(self, transform=None):
        with open('data/train_images.txt') as f:
            self.train_image_paths = [l for l in f.read().splitlines()]

        with open('data/train_boxes.txt') as f:
            self.train_image_boxes = [l for l in f.read().splitlines()]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        im = Image.open('./data/images/' + self.train_image_paths[index]).convert('RGB')
        image = self.transform(im)
        boxes = box_transform(self.train_image_boxes[index], im.size).transform()
        return image, boxes

    def __len__(self):
        return len(self.train_image_paths)


class ILOTestDataset(Dataset):
    def __init__(self, transform=None):
        with open('data/test_images.txt') as f:
            self.test_image_paths = [l for l in f.read().splitlines()]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        im = Image.open('./data/images/' + self.test_image_paths[index]).convert('RGB')
        image = self.transform(im)
        return image

    def __len__(self):
        return len(self.test_image_paths)


train_dataset = ILODataset()
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE
)

batch = next(iter(train_dataloader))
images, boxes = batch

# show_images()
# print(boxes[0].shape)
## create model train and test


model = models.resnet18(pretrained=True)

fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)

model = train_network(model)

test_dataset = ILOTestDataset()
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE
)


images = next(iter(test_dataloader))
show_images(images)
outputs = model(images)



print(outputs)

test_single_image()