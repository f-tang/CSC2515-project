import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

from torchvision.utils import make_grid, save_image
from skimage.color import lab2rgb
from skimage import io

from neural_networks import nn_classification, ColorNet
from myimgfolder import ValImageFolder

import numpy as np
import matplotlib.pyplot as plt


USE_CUDA = torch.cuda.is_available()


def test_output():
    BATCH_SIZE = 1
    DATA_DIR = "../CSC2515_data/cifar/test/"
    scale_transform = transforms.Compose([
        transforms.Scale(32),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    test_set = ValImageFolder(
        root=DATA_DIR, transform=scale_transform)
    test_set_size = len(test_set)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )

    # (img_original, img_scale), y = iter(test_loader).next()

    color_model = ColorNet()
    color_model.load_state_dict(torch.load('colornet_params.pkl'))
    if USE_CUDA:
        color_model.cuda()
    color_model.eval()

    i_gray = 0
    i_color = 0
    i_original = 0
    count = 0
    for data, label in test_loader:
        gray_img = data[0].unsqueeze(1).float()

        # gray_name = '../CSC2515_output/gray/' + str(i_gray) + '.jpg'
        # for img in gray_img:
        #     pic = img.squeeze().numpy()
        #     pic = pic.astype(np.float64)
        #     plt.imsave(gray_name, pic, cmap='gray')
        #     i_gray += 1

        w = gray_img.size()[2]
        h = gray_img.size()[3]
        scale_img = data[1].unsqueeze(1).float()
        if USE_CUDA:
            gray_img, scale_img = gray_img.cuda(), scale_img.cuda()

        gray_img, scale_img = Variable(gray_img, volatile=True), Variable(scale_img)
        pred_label, output = color_model(gray_img, scale_img)
        color_img = torch.cat((gray_img, output[:, :, 0:w, 0:h]), 1)
        color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
        for img in color_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            color_name = '../CSC2515_output/colorimg/' + str(i_color) + '.jpg'
            plt.imsave(color_name, img)
            i_color += 1

        original_img = data[2].float().numpy()
        original_name = '../CSC2515_output/groundtruth/' + str(i_original) + '.jpg'
        for img in original_img:
            pic = img.astype(np.float64)
            plt.imsave(original_name, pic)
            i_original += 1

        # y = label.type(torch.IntTensor).numpy().squeeze()
        # pred_y = pred_label.type(torch.IntTensor).numpy().squeeze()
        # for j in range(BATCH_SIZE):
        #     print("Predicted: %s \t Truth: %s"
        #         %(classes[pred_y[j]], classes[y[j]]))
        #     if pred_y[j] == y[j]:
        #         count += 1


def test_trainset():
    BATCH_SIZE = 5
    DATA_DIR = "../CSC2515_data/cifar/test/"
    scale_transform = transforms.Compose([
        transforms.Scale(32),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    test_set = ValImageFolder(
        root=DATA_DIR, transform=scale_transform)
    test_set_size = len(test_set)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )

    # (img_original, img_scale), y = iter(test_loader).next()

    color_model = ColorNet()
    color_model.load_state_dict(torch.load('colornet_params.pkl'))
    if USE_CUDA:
        color_model.cuda()
    color_model.eval()

    data, label = iter(test_loader).next()
    gray_img = data[0].unsqueeze(1).float()

    fig = plt.figure()
    i = 1
    for img in gray_img:
        pic = img.squeeze().numpy()
        pic = pic.astype(np.float64)
        fig.add_subplot(3, 5, i)
        i += 1
        plt.imshow(pic, cmap='gray')


    w = gray_img.size()[2]
    h = gray_img.size()[3]
    scale_img = data[1].unsqueeze(1).float()
    if USE_CUDA:
        gray_img, scale_img = gray_img.cuda(), scale_img.cuda()

    gray_img, scale_img = Variable(gray_img, volatile=True), Variable(scale_img)
    pred_label, output = color_model(gray_img, scale_img)
    color_img = torch.cat((gray_img, output[:, :, 0:w, 0:h]), 1)
    color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
    for img in color_img:
        img[:, :, 0:1] = img[:, :, 0:1] * 100
        img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
        img = img.astype(np.float64)
        img = lab2rgb(img)
        fig.add_subplot(3, 5, i)
        i += 1
        plt.imshow(img)

    original_img = data[2].float().squeeze().numpy()
    for img in original_img:
        # pic = img.squeeze().numpy()
        pic = img.astype(np.float64)
        fig.add_subplot(3, 5, i)
        i += 1
        plt.imshow(pic)

    plt.show()


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def test_classification():
    DATA_ROOT = "../CSC2515_data/"
    BATCH_SIZE = 4
    LR = 0.001
    MOMENTUM = 0.9
    EPOCH = 2
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    test_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False,
        download=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2
    )

    nn_cl = torch.load('nn_classification.pkl')
    nn_cl.cpu()

    # x_np = test_set.test_data[0]
    # x = torch.from_numpy(x_np)
    # test_x = Variable(x).type(torch.FloatTensor).cuda()
    x, y = iter(test_loader).next()
    y = y.type(torch.IntTensor).numpy().squeeze()
    test_x = Variable(x).cpu()
    test_output = nn_cl(test_x)
    _, pred_y = torch.max(test_output.data, 1)
    pred_y = pred_y.type(torch.IntTensor).numpy().squeeze()
    # print(pred_y[1])

    imshow(torchvision.utils.make_grid(x))
    plt.show()
    print('GroundTruth: ', ' '.join('%5s' % classes[y[j]] for j in range(4)))
    print('Predicted: ', ' '.join('%5s' % classes[pred_y[j]]
                                  for j in range(4)))


if __name__ == '__main__':
    # test_classification()
    test_output()
    # test_trainset()