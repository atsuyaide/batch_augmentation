# reference
# https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.TenCrop
# https://gist.github.com/amirhfarzaneh/66251288d07c67f6cfd23efc3c1143ad

import torch
import numpy as np
import numbers
import random
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn.functional as NF
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

try:
    import accimage
except ImportError:
    accimage = None

import torch.nn as nn


class Net(nn.Module):

    def __init__(self, in_channels=3, flatten_shape=200, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(flatten_shape, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = NF.relu(self.conv1(x))
        x = self.pool(NF.relu(self.conv2(x)))
        x = self.pool(NF.relu(self.conv2(x)))
        x = self.fc1(x.view(batch_size, -1))
        return x

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


class NRandomCrop(object):

    def __init__(self, size, n=1, padding=0, h_flip=False, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.h_flip = h_flip
        self.pad_if_needed = pad_if_needed
        self.n = n

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for i in range(n)]
        j_list = [random.randint(0, w - tw) for i in range(n)]
        return i_list, j_list, th, tw

    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.n)

        return n_random_crops(img, i, j, h, w, self.h_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def n_random_crops(img, x, y, h, w, h_flip):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    crops = []
    for i in range(len(x)):
        new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        if h_flip and random.uniform(0, 1.0) < .5:
            new_crop = new_crop.transpose(Image.FLIP_LEFT_RIGHT)

        crops.append(new_crop)
    return tuple(crops)


def main():
    lr = .1
    momentum = .9
    weight_decay = 5e-4
    batch_size = 64
    n = 8
    pad = 4
    hflip = True
    size = 32
    epochs = 100

    normalize = transforms.Normalize((.5, 0.5, 0.5), (.5, 0.5, 0.5))

    train_transform = transforms.Compose([NRandomCrop(size, n=n, padding=pad, h_flip=hflip),  # this is a list of PIL Images
                         transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
                         transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                         ])
    trainset = CIFAR10(root="./data/", train=True,
                       download=True, transform=train_transform)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        print("-----------------")
        for batch in trainloader:
            # In your test loop you can do the following:
            input, target = batch  # input is a 5d tensor, target is 2d
            bs, ncrops, c, h, w = input.size()

            optimizer.zero_grad()
            result = net(input.view(-1, c, h, w))  # fuse batch size and ncrops
            target = np.repeat(target, ncrops)

            loss = criterion(result, target)
            loss.backward()
            print(loss)
            optimizer.step()

if __name__ == "__main__":
    main()
