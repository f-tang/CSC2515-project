import torchvision
from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

scale_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    #transforms.ToTensor()
])


class TrainImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img_original = img
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = img_original.numpy()

        img_original = img_original.transpose((1, 2, 0))

        img_lab = rgb2lab(img_original)
        img_lab = (img_lab + 128) / 255
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))

        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img_original, img_ab), target


class ValImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img_original = img
        img_scale = img.copy()
        if self.transform is not None:
            img_scale = self.transform(img_scale)
            img_scale = img_scale.numpy()
            img_original = self.transform(img_original)
            img_original = img_original.numpy()

        img_original = img_original.transpose((1, 2, 0))
        img_scale = img_scale.transpose((1, 2, 0))

        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        img_gray = rgb2gray(img_original)
        img_gray = torch.from_numpy(img_gray)
        img_original = torch.from_numpy(img_original)

        return (img_gray, img_scale, img_original), target


def read_data():
    DATA_ROOT = "../CSC2515_data/"
    TRAIN_DATA_ROOT = "../CSC2515_data/cifar/train/"
    TEST_DATA_ROOT = "../CSC2515_data/cifar/test/"
    BATCH_SIZE = 4
    LR = 0.001
    MOMENTUM = 0.9
    EPOCH = 2
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_gray = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_set_gray = TrainImageFolder(
        root=TRAIN_DATA_ROOT, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set_gray, batch_size=32, shuffle=True, num_workers=4)
    print(train_set_gray.classes)

    (img_gray,img_ab), y = iter(train_loader).next()
    img = torchvision.utils.make_grid(img_gray)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg[0], cmap='gray')
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


    # for batch_idx, (data, classes) in enumerate(train_loader):
    #     original_img = data[0].unsqueeze(1).float()
    #     img_ab = data[1].float()
    #     print(original_img)
    #     print(img_ab)


if __name__ == '__main__':
    read_data()