import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from neural_networks import nn_classification, ColorNet
from myimgfolder import TrainImageFolder

import os
import traceback
import time


USE_CUDA = torch.cuda.is_available()


def calculate_mse_loss(test_y, pred_y):
    diff_pow = torch.pow((test_y - pred_y), 2)
    pow_sum = diff_pow.sum()
    pred_size = pred_y.size()
    pred_size = torch.from_numpy(np.array(list(pred_size)))
    size_prod = pred_size.prod()
    res = pow_sum / size_prod

    return res


def train():
    LR = 1.0
    EPOCHS = 6
    BATCH_SIZE = 32
    DATA_DIR = "../CSC2515_data/cifar/train/"
    original_transform = transforms.Compose([
        transforms.Scale(32),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_set = TrainImageFolder(
        root=DATA_DIR, transform=original_transform
    )
    train_set_size = len(train_set)
    train_set_classes = train_set.classes
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    # DEBUG:
    (img_gray, img_lab), y = iter(train_loader).next()
    img_gray = img_gray.numpy()
    img_lab = img_lab.numpy()

    color_model = ColorNet()
    if os.path.exists('./colornet_params.pkl'):
        color_model.load_state_dict(torch.load('colornet_params.pkl'))
    if USE_CUDA:
        color_model.cuda()
    optimizer = torch.optim.Adadelta(color_model.parameters())

    print("start training")

    train_time = 0
    for epoch in range(EPOCHS):
        start_time = int(round(time.time()))
        color_model.train()

        try:
            for batch_idx, (data, classes) in enumerate(train_loader):
                messagefile = open('./message.log', 'a')
                original_img = data[0].unsqueeze(1).float()
                img_ab = data[1].float()

                if USE_CUDA:
                    original_img = original_img.cuda()
                    img_ab = img_ab.cuda()
                    classes = classes.cuda()

                original_img = Variable(original_img)
                img_ab = Variable(img_ab)
                classes = Variable(classes)

                optimizer.zero_grad()

                class_output, output = color_model(original_img, original_img)
                # mse_func = nn.MSELoss()
                mse_loss = calculate_mse_loss(img_ab, output)
                mse_loss.backward(retain_variables=True)
                # l1_func = nn.L1Loss()
                # l1_loss = l1_func(output, img_ab)
                # l1_loss.backward(retain_variables=True)
                # sl1_func = nn.SmoothL1Loss()
                # sl1_loss = sl1_func(output, img_ab)
                # sl1_loss.backward(retain_variables=True)
                cross_entropy_loss = 0.001 * F.cross_entropy(class_output, classes)
                cross_entropy_loss.backward()

                optimizer.step()

                loss = mse_loss.data[0] + cross_entropy_loss
                lossmsg = 'loss: %.9f\n' % (loss.data[0])
                messagefile.write(lossmsg)

                if batch_idx % 500 == 0:
                    message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                        epoch, batch_idx * BATCH_SIZE, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0])
                    print(message)
                    messagefile.write(message)
                    torch.save(color_model.state_dict(), 'colornet_params.pkl')
                messagefile.close()

        except Exception:
            print("an error occurs")
            logfile = open('error.log', 'w')
            logfile.write(traceback.format_exc())
            logfile.close()
        finally:
            torch.save(color_model.state_dict(), 'colornet_params.pkl')

        end_time = int(round(time.time()))
        time_interval = end_time - start_time
        print("train time for epoch %d: %d" % (epoch, time_interval))
        print("training speed: %f s/pic"
                          % (time_interval/len(train_loader.dataset)))
        messagefile = open('./message.log', 'a')
        messagefile.write("train time for epoch %d: %d\n" %(epoch, time_interval))
        messagefile.write("training speed: %f s/pic\n"
                          %(time_interval/len(train_loader.dataset)))
        messagefile.close()

    print("training finished")



def train_classification():
    DATA_ROOT = "../CSC2515_data/"
    BATCH_SIZE = 4
    LR = 0.0001
    MOMENTUM = 0.9
    EPOCH = 2
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=False,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False,
        download=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2
    )

    train_set_np = train_set.train_data[0]
    print(train_set_np)

    nn_cl = nn_classification()
    nn_cl.cuda()
    optimizer = torch.optim.SGD(nn_cl.parameters(), lr=LR, momentum=MOMENTUM)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            b_x, b_y = Variable(x).cuda(), Variable(y).cuda()

            optimizer.zero_grad()

            outputs = nn_cl(b_x)
            loss = loss_func(outputs, b_y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if step % 2000 == 1999:  # print every 2000 mini-batches
                print('epoch: %d, step: %5d, loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.0

    print("Training Finished")
    torch.save(nn_cl, 'nn_classification.pkl')  # entire net
    torch.save(nn_cl.state_dict(), 'nn_classification_params.pkl')  # parameters



if __name__ == '__main__':
    # train_classification()
    train()