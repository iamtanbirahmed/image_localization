import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patches
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import utils as Util


def load_model():
    image_localization_model = torch.load('model.pt')
    return image_localization_model


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
    # 137.60233 71.958336 242.84622 280.39297

    def nearest_square_numebr(limit=111):
        answer = 0
        while (answer + 1) ** 2 < limit:
            answer += 1
        return answer ** 2

    with open('./data/test_images.txt') as f:
        test_image_path = [l.strip() for l in f.read().splitlines()]
    with open('output.txt') as f:
        test_image_box = [l.strip() for l in f.read().splitlines()]

    fig = plt.figure(figsize=(20, 20))

    total_test_image = len(test_image_path)
    square_matrix_dimention = np.math.sqrt(nearest_square_numebr(total_test_image)) + 1

    for i in range(total_test_image):
        # fig, ax = plt.subplots(1)
        im = Image.open('./data/images/' + test_image_path[i]).convert('RGB')

        im = np.asarray(im)
        ax = fig.add_subplot(square_matrix_dimention, square_matrix_dimention, i + 1)
        ax.imshow(im)

        dimentions = test_image_box[i].split(' ')
        dimentions = list(map(float, dimentions))
        x, y, w, h = dimentions
        rect_pred = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r',
                                      facecolor='none')
        # rect_original = patches.Rectangle((x - 10, y - 10), w - 10, h - 10, linewidth=1, edgecolor='g',
        #                                   facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect_pred)
        # ax.add_patch(rect_original)

    plt.show()


test_dataset = ILOTestDataset()
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=10
)


# testing the dataloader

def test_initialization():
    batch, im_size = next(iter(test_dataloader))
    images = batch

    model = load_model()
    output = model(images)

    reversed_box = Util.box_transform_inv(output, im_size)


def test_network(model):
    FILE_NAME = 'output.txt'
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


model = load_model()
test_network(model)

test_single_image()
