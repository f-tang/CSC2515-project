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
BATCH_SIZE = 1


def test():
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

    i = 0
    count = 0
    for data, label in test_loader:
        original_img = data[0].unsqueeze(1).float()
        gray_name = '../CSC2515_output/gray/' + str(i) + '.jpg'
        for img in original_img:
            pic = img.squeeze().numpy()
            pic = pic.astype(np.float64)
            plt.imsave(gray_name, pic, cmap='gray')
        w = original_img.size()[2]
        h = original_img.size()[3]
        scale_img = data[1].unsqueeze(1).float()
        if USE_CUDA:
            original_img, scale_img = original_img.cuda(), scale_img.cuda()

        original_img, scale_img = Variable(original_img, volatile=True), Variable(scale_img)
        pred_label, output = color_model(original_img, scale_img)
        color_img = torch.cat((original_img, output[:, :, 0:w, 0:h]), 1)
        color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
        for img in color_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            color_name = '../CSC2515_output/colorimg/' + str(i) + '.jpg'
            plt.imsave(color_name, img)
            i += 1

        # y = label.type(torch.IntTensor).numpy().squeeze()
        # pred_y = pred_label.type(torch.IntTensor).numpy().squeeze()
        # for j in range(BATCH_SIZE):
        #     print("Predicted: %s \t Truth: %s"
        #         %(classes[pred_y[j]], classes[y[j]]))
        #     if pred_y[j] == y[j]:
        #         count += 1

    accuracy = float(count) / float(test_set_size)
    print("test finished, accuracy: %f" %(accuracy))




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
    test()