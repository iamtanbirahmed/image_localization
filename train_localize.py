import numpy as np
import progressbar
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models

import data.utils as Utils

BATCH_SIZE = 32


class ILODataset(Dataset):
    def __init__(self, transform=None):
        with open('data/train_images.txt') as f:
            self.train_image_paths = [l for l in f.read().splitlines()]

        with open('data/train_boxes.txt') as f:
            self.train_image_boxes = [l for l in f.read().splitlines()]

        if transform is None:
            self.transform = transforms.Compose([
                # transforms.RandomVerticalFlip(),
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        im = Image.open('./data/images/' + self.train_image_paths[index]).convert('RGB')
        image = self.transform(im)
        # boxes = box_transform(self.train_image_boxes[index], im.size).transform()
        boxes = list(map(float, self.train_image_boxes[index].split(' ')))
        boxes = np.array(boxes, dtype='float32')
        im_size = np.array(im.size, dtype='float32')
        return image, boxes, im_size

    def __len__(self):
        return len(self.train_image_paths)


train_dataset = ILODataset()
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10
)

# testing the dataloader
batch = next(iter(train_dataloader))
images, boxes, im_size = batch


def train_network(model):
    EPOCHS = 20
    running_loss = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.SmoothL1Loss()  ## the loss function

    for epoch in range(EPOCHS):
        bar = progressbar.ProgressBar()
        max_acc = 0
        max_acc_over_all_epoc = 0
        for i, data in enumerate(train_dataloader, 0):

            images, boxes, im_size = data
            inputs = images
            targets = Utils.box_transform(boxes, im_size)
            # targets = boxes
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            acc = Utils.compute_acc(outputs, targets, im_size)
            if acc > max_acc:
                max_acc = acc
            print("Epoch {}/{} of {}, Loss: {:.3f}, Accuracy: {:.3f}".format(i, epoch + 1, EPOCHS, loss.item(), acc))

        print("Epoch {}/{}, Max Accuracy: {:.3f}".format(epoch + 1, EPOCHS, max_acc))
        if max_acc > max_acc_over_all_epoc:
            max_acc_over_all_epoc = max_acc

    print('Finished Training. Maximum Accuracy:{}'.format(max_acc_over_all_epoc))
    return model


## create model train and test


model = models.resnet18(pretrained=True)

fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)
model = train_network(model)

torch.save(model,'model.pt')

