# 10/31/2017 Halloween Day Init
# Author: Zhi Huang
# Purdue University
# IU Center for Neuroimaging

import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import dataset.kitti
import dataset.dental
#from dataset import KittiRoadTrain  # NOQA
#from dataset import KittiRoadValidate
import numpy as np

number_of_class = 4

def happyprint(string, obj):
    # print(string, obj)
    return

parser = argparse.ArgumentParser(description='FCN 2D')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default=1)')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default=1)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='input number of epochs to train')
parser.add_argument('--lr', type=float, default=1.0e-10, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

root = '/media/zhi/Drive3/KITTI/rwth_kitti_semantics_dataset'
root = '/media/zhi/Drive3/Dental_Analysis/imagelabeler/dataset_downsampling2'

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(dataset.DentalClassSeg(root, split='train', 
              transform=True), batch_size = args.batch_size, shuffle=True, **kwargs)
validation_loader = torch.utils.data.DataLoader(dataset.DentalClassSegValidate(root, split='validation',
                transform=True), batch_size = args.test_batch_size, shuffle=False, **kwargs)

class Net(nn.Module):
    def __init__(self, n_class = number_of_class):
        super().__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    # def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
    #     """Make a 2D bilinear kernel suitable for upsampling"""
    #     factor = (kernel_size + 1) // 2
    #     if kernel_size % 2 == 1:
    #         center = factor - 1
    #     else:
    #         center = factor - 0.5
    #     og = np.ogrid[:kernel_size, :kernel_size]
    #     filt = (1 - abs(og[0] - center) / factor) * \
    #            (1 - abs(og[1] - center) / factor)
    #     weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
    #                       dtype=np.float64)
    #     weight[range(in_channels), range(out_channels), :, :] = filt
    #     return torch.from_numpy(weight).float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                m.weight.data.normal_(0.0, 0.06)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                print()
                m.weight.data.normal_(0.0, 0.06)
                # assert m.kernel_size[0] == m.kernel_size[1]
                # initial_weight = self.get_upsampling_weight(
                #     m.in_channels, m.out_channels, m.kernel_size[0])
                # m.weight.data.copy_(initial_weight)


    def forward(self, x):
        h = x
        happyprint("init: ", x.data[0].shape)

        h = self.relu1_1(self.conv1_1(h))
        happyprint("after conv1_1: ", h.data[0].shape)
        
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        happyprint("after pool1: ", h.data[0].shape)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        happyprint("after pool2: ", h.data[0].shape)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        happyprint("after pool3: ", h.data[0].shape)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        happyprint("after pool4: ", h.data[0].shape)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        happyprint("after pool5: ", h.data[0].shape)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        happyprint("after score_fr: ", h.data[0].shape)
        h = self.upscore2(h)

        happyprint("after upscore2: ", h.data[0].shape)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        happyprint("after score_pool4: ", h.data[0].shape)

        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        happyprint("after score_pool4c: ", h.data[0].shape)

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        happyprint("after upscore_pool4: ", h.data[0].shape)

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        happyprint("after score_pool3: ", h.data[0].shape)

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        happyprint("after upscore8: ", h.data[0].shape)

        return F.log_softmax(h)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # plt.imshow(data[0,0,:,:].cpu().numpy())
        # plt.title('data')
        # plt.pause(0.3)
        # plt.imshow(target[0,:,:].cpu().numpy())
        # plt.title('target')
        # plt.pause(0.3)

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        happyprint("data shape: ", data.size())
        happyprint("target shape: ", target.size())
        output = model(data)

        # plt.imshow(output[0,0,:,:].data.cpu().numpy())
        # plt.title('output (score)')
        # plt.pause(0.3)

        n, c, h, w = output.size()

        happyprint("log max: ", output.max())
        happyprint("log min: ", output.min())
        happyprint("output shape 1: ", output.size())

        output = output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        output = output.view(-1, c)
        mask = target >= 0
        target = target[mask]

        happyprint("target max: ", target.max())
        happyprint("target min: ", target.min())
        happyprint("output shape 2: ", output.size())
        happyprint("target shape: ", target.size())

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        print("data dim", data.size())
        print("target dim", target.size())
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs+1):
    train(epoch)
    test()