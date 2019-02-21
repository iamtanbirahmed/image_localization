import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models

import utils as Utils

BATCH_SIZE = 32


def fine_tune_model():
    pretrained_resnet_18_model = models.resnet18(pretrained=True)
    fc_in_size = pretrained_resnet_18_model.fc.in_features
    pretrained_resnet_18_model.fc = nn.Linear(fc_in_size, 4)
    return pretrained_resnet_18_model


class ILODataset(Dataset):
    def __init__(self, transform=None):
        with open('./data/train_images.txt') as f:
            self.train_image_paths = [l for l in f.read().splitlines()]

        with open('./data/train_boxes.txt') as f:
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
        image_dimentions = np.array(im.size, dtype='float32')
        return image, boxes, image_dimentions

    def __len__(self):
        return len(self.train_image_paths)


train_dataset = ILODataset()
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10
)

# testing the dataloader
batch = next(iter(train_dataloader))
images, boxes, image_dimentions = batch


def plot_result():
    with open('accuracy.txt') as f:
        accuracy = [list(map(float, l.split(' '))) for l in f.read().splitlines()]
    with open('loss.txt') as f:
        loss = [list(map(float, l.split(' '))) for l in f.read().splitlines()]
    x_batches = [item[0] for item in accuracy]
    y_accuracy = [item[1] for item in accuracy]

    x_loss_batches = [item[0] for item in loss]
    y_loss = [item[1] for item in loss]

    plt.subplot(2, 1, 1)
    plt.plot(x_batches, y_accuracy, 'g')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(x_loss_batches, y_loss, 'r')
    plt.xlabel('Batch')
    plt.ylabel('loss')
    plt.show()
    # x_batches = [x for x in results.split(' ')[:,0]]


def train_network(model):
    EPOCHS = 1500
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.SmoothL1Loss()  ## the loss function
    f_acc = open('accuracy.txt', 'w')
    f_loss = open('loss.txt', 'w')
    j = 0
    for epoch in range(EPOCHS):

        max_acc = 0
        max_acc_over_all_epoc = 0
        for i, data in enumerate(train_dataloader, 0):

            images, boxes, image_dimentions = data
            inputs = images
            targets = Utils.box_transform(boxes, image_dimentions)
            # targets = boxes
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            acc = Utils.compute_acc(outputs, targets, image_dimentions)
            if acc > max_acc:
                max_acc = acc
            print("Epoch {}/{} of {}, Loss: {:.3f}, Accuracy: {:.3f}".format(i, epoch + 1, EPOCHS, loss.item(), acc))
            f_acc.write(str(j) + ' ' + str(acc.detach().numpy()) + '\n')
            f_loss.write(str(j) + ' ' + str(loss.item()) + '\n')
            j = j + 1
        print("Epoch {}/{}, Max Accuracy: {:.3f}".format(epoch + 1, EPOCHS, max_acc))
        if max_acc > max_acc_over_all_epoc:
            max_acc_over_all_epoc = max_acc

    print('Finished Training. Maximum Accuracy:{}'.format(max_acc_over_all_epoc))
    f_acc.close()
    f_loss.close()
    return model


## create model train and test


image_localization_model = train_network(fine_tune_model())

# torch.save(model, 'model.pt')
torch.save(image_localization_model, 'model.pt')
plot_result()
