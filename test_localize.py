import torch
from PIL import Image
from matplotlib import patches
from progressbar import progressbar
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import data.utils as Util
import torch.nn as nn

model = torch.load('model.pt')
print(model)


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
        im_size = np.array(im.size, dtype='float32')
        return image, im_size

    def __len__(self):
        return len(self.test_image_paths)


def test_single_image():
    # 88.81446   60.741096 157.93982  257.19263
    # 114.18105 57.434986 259.98145 270.80618
    #137.60233 71.958336 242.84622 280.39297
    with open('./data/test_images.txt') as f:
        test_image_path = [l.trim() for l in f.read().splitlines()]
    with open('./data/test_boxes.txt') as f:
        test_image_box = [l.trim() for l in f.read().splitlines()]

    if len(test_image_box) == len(test_image_path):
        for i in range(10):

            fig, ax = plt.subplots(1)
            im = Image.open('./data/images/' + test_image_path[i]).convert('RGB')
            ts = transforms.Compose([
                # transforms.Scale((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])
            ax.imshow(np.transpose(ts(im), (1, 2, 0)))
            # ax.imshow(np.transpose(im, (1, 2, 0)))

            # Create a Rectangle patch
            print(test_image_box[i].split(' '))
            # rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            # ax.add_patch(rect)

    plt.show()


test_dataset = ILOTestDataset()
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=10
)
# testing the dataloader
batch, im_size = next(iter(test_dataloader))
images = batch

output = model(images)

reversed_box = Util.box_transform_inv(output, im_size)


def test_network(model):
    FILE_NAME = './data/test_boxes.txt'
    f = open(FILE_NAME, 'w')
    EPOCHS = 1
    for epoch in range(EPOCHS):
        for i, data in enumerate(test_dataloader, 0):
            images, im_size = data
            inputs = images
            outputs = model(inputs)
            outputs = Util.box_transform_inv(outputs, im_size)
            outputs = outputs.detach().numpy()[0]
            for value in outputs:
                f.write(str(value) + ' ')
            f.write('\n')

    print('Finished Testing')
    return model


# test_network(model)
test_single_image()
