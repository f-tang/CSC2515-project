import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from neural_networks import nn_classification
import numpy as np


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
    test_classification()