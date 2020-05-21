import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
import os
import h5py


class DoubleConv(nn.Module):
    def __init__(self, In, Out):
        super().__init__()
        self.conv1 = nn.Conv2d(In, Out, 3, 1, 1)
        self.conv2 = nn.Conv2d(Out, Out, 3, 1, 1)
        self.norm = nn.BatchNorm2d(Out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class Down(nn.Module):
    def __init__(self, In, Out):
        super(Down, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.double_conv = DoubleConv(In, Out)

    def forward(self, x):
        x = self.pool1(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, In, Out):
        super(Up, self).__init__()

        self.upsam1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(2 * Out, In)
        self.conv1 = nn.Conv2d(In, Out, 3, 1, 1)
        # self.norm = nn.BatchNorm2d(In)

    def forward(self, x1, x2):
        x1 = self.upsam1(x1)
        # x2 = self.norm(x2)
        x2 = self.conv1(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)

        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.double_conv = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(64, 128)
        self.up2 = Up(32, 64)
        self.conv = nn.Conv2d(32, 1, 3, 1, 1)
        # self.norm = nn.BatchNorm2d(1)

    def forward(self, x1):
        x1 = self.double_conv(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.norm(x)
        x = self.conv(x)
        return x


mode = 0
dataset_PATH = 'C:/Users/Ryuusei/PycharmProjects/XPRIZE/DANV-master/mfcc_test_h5/mfcc_python'
target_PATH = 'C:/Users/Ryuusei/PycharmProjects/XPRIZE/DANV-master/mfcc_test_h5/mfcc_matlab'
model_PATH = 'C:/Users/Ryuusei/PycharmProjects/XPRIZE/DANV-master'

for filename in os.listdir(model_PATH):
    if filename.endswith("model.pth"):
        mode = 1
if mode == 0:
    net = Unet()
    net = net.double()

elif mode == 1:
    net = torch.load('model.pth')

# inputs = np.random.rand(128, 1, 12, 20)
# inputs = torch.from_numpy(inputs)
# inputs = inputs.double()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
x = 0
for epoch in range(10):
    if epoch > 0:
        print('Epoch Change')
    running_loss = 0.0
    i = 0
    for filename in os.listdir(dataset_PATH):
        if filename.endswith('h5'):
            # print(filename)
            i += 1
            with h5py.File(dataset_PATH + '/' + filename, 'r') as input_data:
                a_group_key = list(input_data.keys())[0]
                inputs = input_data[a_group_key]
                # print(inputs.shape)
                inputs = np.array(inputs)
                inputs = np.reshape(inputs, (1025, 1, 12, 20))
                inputs = torch.from_numpy(inputs)
                inputs = inputs.double()
                start = time.time()
                optimizer.zero_grad()
                with h5py.File(target_PATH + '/mfcc_matlab' + str(i) + '.h5', 'r') as target_data:
                    b_group_key = list(target_data.keys())[0]
                    targets = target_data[b_group_key]
                    targets = np.array(targets)
                    targets = np.reshape(targets, (1025, 1, 12, 20))
                    targets = torch.from_numpy(targets)
                    targets = targets.double()
                    trained = net(inputs)
                    loss = nn.MSELoss()
                    output = loss(trained, targets)
                    output.backward()
                    optimizer.step()
                    running_loss += output.item()
                    x += 1
        if x % 10 == 9:  # print every 10 mini-batches
            end = time.time()
            period = end - start
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            print('time/step: %.5f' % period)
        if x % 50 == 49:
            torch.save(net.state_dict(),
                       './model.pth')
            print('Model Updated')
        running_loss = 0.0
    x += 1

print('Finished Training')
