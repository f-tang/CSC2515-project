from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class nn_classification(nn.Module):
    def __init__(self):
        super(nn_classification, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class LowLevelFeatNet(nn.Module):
    def __init__(self):
        super(LowLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)       # 32*32 -> 16*16
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)     # 16*16 -> 16*16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)    # 16*16 -> 8*8
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        # x1 = F.relu(self.bn5(self.conv5(x1)))
        x1 = F.relu(self.bn6(self.conv6(x1)))
        if self.training:
            x2 = x1.clone()
        else:
            x2 = F.relu(self.bn1(self.conv1(x2)))
            x2 = F.relu(self.bn2(self.conv2(x2)))
            x2 = F.relu(self.bn3(self.conv3(x2)))
            x2 = F.relu(self.bn4(self.conv4(x2)))
            # x2 = F.relu(self.bn5(self.conv5(x2)))
            x2 = F.relu(self.bn6(self.conv6(x2)))
        return x1, x2


class MidLevelFeatNet(nn.Module):
    def __init__(self):
        super(MidLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class GlobalFeatNet(nn.Module):
    def __init__(self):
        super(GlobalFeatNet, self).__init__()
        # self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(512)
        # self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(512)
        # self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(8*8*512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 8*8*512)
        x = F.relu(self.bn5(self.fc1(x)))
        output_512 = F.relu(self.bn6(self.fc2(x)))
        output_256 = F.relu(self.bn7(self.fc3(output_512)))
        return output_512, output_256


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.bn4 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.log_softmax(self.bn4(self.fc4(x)))
        return x


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, mid_input, global_input):
        w = mid_input.size()[2]
        h = mid_input.size()[3]
        global_input = global_input.unsqueeze(2).unsqueeze(2).expand_as(mid_input)
        fusion_layer = torch.cat((mid_input, global_input), 1)
        fusion_layer = fusion_layer.permute(2, 3, 0, 1).contiguous()
        fusion_layer = fusion_layer.view(-1, 512)
        fusion_layer = self.bn1(self.fc1(fusion_layer))
        fusion_layer = fusion_layer.view(w, h, -1, 256)

        x = fusion_layer.permute(2, 3, 0, 1).contiguous()
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        # x = self.upsample(x)
        x = F.sigmoid(self.bn5(self.conv4(x)))
        x = self.upsample(self.conv5(x))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.low_lv_feat_net = LowLevelFeatNet()
        self.mid_lv_feat_net = MidLevelFeatNet()
        self.global_feat_net = GlobalFeatNet()
        self.class_net = ClassificationNet()
        self.upsample_col_net = ColorizationNet()

    def forward(self, x1, x2):
        x1, x2 = self.low_lv_feat_net(x1, x2)
        #print('after low_lv, mid_input is:{}, global_input is:{}'.format(x1.size(), x2.size()))
        x1 = self.mid_lv_feat_net(x1)
        #print('after mid_lv, mid2fusion_input is:{}'.format(x1.size()))
        class_input, x2 = self.global_feat_net(x2)
        #print('after global_lv, class_input is:{}, global2fusion_input is:{}'.format(class_input.size(), x2.size()))
        class_output = self.class_net(class_input)
        #print('after class_lv, class_output is:{}'.format(class_output.size()))
        output = self.upsample_col_net(x1, x2)
        #print('after upsample_lv, output is:{}'.format(output.size()))
        return class_output, output