import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from neural_networks import nn_classification

def train_classification():
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
    train_classification()