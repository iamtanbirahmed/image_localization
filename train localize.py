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


# grid = torchvision.utils.make_grid(images, nrow=10)
# plt.figure(figsize=(50, 50))
# plt.imshow(np.transpose(grid, (1, 2, 0)))
# plt.show()


class ILODataset(Dataset):
    def __init__(self):
        with open('data/train_images.txt') as f:
            self.train_image_paths = [l for l in f.read().splitlines()]

        with open('data/train_boxes.txt') as f:
            self.train_image_boxes = [l for l in f.read().splitlines()]

        self.transform = transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

    def __getitem__(self, index):
        im = Image.open('./data/images/' + self.train_image_paths[index]).convert('RGB')
        image = self.transform(im),
        boxes = transform_box_location(self.train_image_boxes[index].split(' '),
                                       tensor_to_nd_array(to_2d_tensor(im.size)))

        return image, to_2d_tensor(im.size), boxes

    def __len__(self):
        return len(self.train_image_paths)


dataset = ILODataset()
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1
)

model = models.resnet18(pretrained=True)

fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.SmoothL1Loss()  ## the loss function

# batch = next(iter(dataloader))
# images, image_sizes, boxes = batch
# plot_images(images, image_sizes, boxes)
# plot_test_Image()

EPOCHS = 20
running_loss = 0.0
batch = 0;
for epoch in range(3):
    bar = progressbar.ProgressBar()
    for images, image_sizes, boxes in bar(dataloader):
        print('[%d batch %d epoch] running..' % (batch, epoch))
        inputs = images
        targets = boxes
        # forward + backward + optimize
        outputs = model(inputs[0])
        loss = loss_func(outputs, targets[0].float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch % 10 == 0:  # print every 1 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch, batch + 1, running_loss))
            running_loss = 0.0
        batch = batch + 1

print('Finished Training')
