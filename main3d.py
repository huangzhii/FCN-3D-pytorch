# 11/08/2017
# Author: Zhi Huang
# Purdue University
# IU Center for Neuroimaging

import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import dataset.kitti
import dataset.dental
import os.path as osp
import gdown
import hashlib



#from dataset import KittiRoadTrain  # NOQA
#from dataset import KittiRoadValidate
import numpy as np

number_of_class = 2

def happyprint(string, obj):
    # print(string, obj)
    return

parser = argparse.ArgumentParser(description='FCN 2D')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default=1)')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default=1)')
parser.add_argument('--epochs', type=int, default=1000000, metavar='N',
                    help='input number of epochs to train')
parser.add_argument('--lr', type=float, default=1.0e-10, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

root = '/media/zhi/Drive3/KITTI/rwth_kitti_semantics_dataset'
root = '/media/zhi/Drive3/Dental_Analysis/imagelabeler/dataset_downsampling2'

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(dataset.Data3D(root, split='train', 
              transform=True), batch_size = args.batch_size, shuffle=True, **kwargs)
validation_loader = torch.utils.data.DataLoader(dataset.Data3DVal(root, split='validation',
                transform=True), batch_size = args.test_batch_size, shuffle=False, **kwargs)



def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_file = _get_vgg16_pretrained_model()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def _get_vgg16_pretrained_model():
    return cached_download(
        url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
        path=osp.expanduser('./vgg16_from_caffe.pth'),
        md5='aa75b158f4181e7f6230029eb96c1b13',
    )
def cached_download(url, path, md5=None, quiet=False):

    def check_md5(path, md5):
        print('[{:s}] Checking md5 ({:s})'.format(path, md5))
        return md5sum(path) == md5

    if osp.exists(path) and not md5:
        print('[{:s}] File exists ({:s})'.format(path, md5sum(path)))
        return path
    elif osp.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        dirpath = osp.dirname(path)
        if not osp.exists(dirpath):
            os.makedirs(dirpath)
        return gdown.download(url, path, quiet=quiet)
def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b''):
            hash.update(block)
    return hash.hexdigest()


class Net(nn.Module):
    def __init__(self, n_class = number_of_class):
        super().__init__()
        # conv1
        self.conv1_1 = nn.Conv3d(1, 8, 3, padding=60)
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=15)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv3d(64, 512, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout3d()

        # fc7
        self.fc7 = nn.Conv3d(512, 512, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout3d()

        self.score_fr = nn.Conv3d(512, n_class, 1)
        self.score_pool3 = nn.Conv3d(32, n_class, 1)
        self.score_pool4 = nn.Conv3d(64, n_class, 1)

        self.upscore2 = nn.ConvTranspose3d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose3d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose3d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()


    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :, :] = filt
        return torch.from_numpy(weight).float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.zero_()
                m.weight.data.normal_(0.0, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose3d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             m.weight.data.zero_()
    #             m.weight.data.normal_(0.0, 0.1)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #                 m.bias.data.normal_(0.0, 0.1)
    #         if isinstance(m, nn.ConvTranspose3d):
    #             m.weight.data.zero_()
    #             m.weight.data.normal_(0.0, 0.1)
    #             # assert m.kernel_size[0] == m.kernel_size[1]
    #             # initial_weight = self.get_upsampling_weight(
    #             #     m.in_channels, m.out_channels, m.kernel_size[0])
    #             # m.weight.data.copy_(initial_weight)

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            print("what is l1? ", l1)
            print("what is l2? ", l2)
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

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

        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3], 5:5 + upscore2.size()[4]]

        score_pool4c = h  # 1/16
        happyprint("after score_pool4c: ", h.data[0].shape)

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        happyprint("after upscore_pool4: ", h.data[0].shape)

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
             9:9 + upscore_pool4.size()[2],
             9:9 + upscore_pool4.size()[3],
             9:9 + upscore_pool4.size()[4]]
        score_pool3c = h  # 1/8
        happyprint("after score_pool3: ", h.data[0].shape)

        # print(upscore_pool4.data[0].shape)
        # print(score_pool3c.data[0].shape)

        # Adjusting stride in self.upscore2 and self.upscore_pool4
        # and self.conv1_1 can change the tensor shape (size).
        # I don't know why!

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h) # dim: 88^3
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3], 31:31 + x.size()[4]].contiguous()
        happyprint("after upscore8: ", h.data[0].shape)
        return h


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
        output_score = model(data)

        # plt.imshow(output[0,0,:,:].data.cpu().numpy())
        # plt.title('output (score)')
        # plt.pause(0.3)

        n, c, h, w, d = output_score.size()

        output = output_score.contiguous().view(-1, c, h, w*d) # h: 60 w*d: 1200
        m = torch.nn.LogSoftmax()
        log_p = m(output)
        happyprint("         maximum val of log_p:", log_p.data.max())
        happyprint("         minimum val of log_p:", log_p.data.min())
        happyprint("output shape 1: ", output.size())

        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p.view(-1, c)
        mask = target >= 0
        target = target[mask]
        happyprint("log_p shape 2: ", log_p.size())
        happyprint("target shape: ", target.size())

        loss = F.nll_loss(log_p, target)
        happyprint("epoch    ----------------------: ", epoch)
        happyprint("batch_idx----------------------: ", batch_idx)
        happyprint("loss     ----------------------: ", loss.data[0])

        if epoch % args.log_interval == 0 and batch_idx == 0:
            #plt.imshow(output_score[0,0,:,:,30].data.cpu().numpy())
            #plt.title('output (score)')
            #plt.pause(0.3)
            imgname = 'epoch' + str(epoch) + 'loss_' + str(loss.data[0]) + '.png'
            plt.imsave(imgname,output_score[0,0,:,:,30].data.cpu().numpy())


        loss.backward()
        optimizer.step()

        if batch_idx % 26 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], output_score[0,0,:,:,30].data.max(), output_score[0,0,:,:,30].data.min()))

def test():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        happyprint("data dim", data.size())
        happyprint("target dim", target.size())
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(validation_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))



start_epoch = 0
start_iteration = 0
vgg16 = VGG16(pretrained=True)

model = Net()
# model.copy_params_from_vgg16(vgg16)
if args.cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
    eps=1e-8, weight_decay=0)


for epoch in range(1, args.epochs+1):
    print("epoch: ",epoch)
    model.train()
    train(epoch)
    # test()