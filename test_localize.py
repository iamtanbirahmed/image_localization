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
    # 31.1151, 67.1595, 219.5780, 266.6911
    fig, ax = plt.subplots(1)

    im = Image.open('./data/images/' + '001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.jpg').convert('RGB')

    ts = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    ax.imshow(np.transpose(ts(im), (1, 2, 0)))
    # ax.imshow(np.transpose(im, (1, 2, 0)))

    # Create a Rectangle patch
    rect = patches.Rectangle((31.1151, 67.1595), 219.5780, 266.6911, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


test_dataset = ILOTestDataset()
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=10
)
# testing the dataloader
batch , im_size = next(iter(test_dataloader))
images = batch

output = model(images)

reversed_box = Util.box_transform_inv(output,im_size)




def test_network(model):
    results = np.empty()
    FILE_NAME = './data/test_boxes.txt'
    f = open('./data/test_boxes.txt','w')
    EPOCHS = 1
    for epoch in range(EPOCHS):
        for i, data in enumerate(test_dataloader, 0):
            images, im_size = data
            inputs = images
            outputs = model(inputs)
            outputs = Util.box_transform_inv(outputs,im_size)

            # outputs = outputs.detach().numpy()
            np.append(results,outputs)

    np.savetxt(FILE_NAME, results, fmt="%.5f")
    print('Finished Testing')
    f.close()
    return model


test_network(model)
