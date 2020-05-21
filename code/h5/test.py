import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import time
import h5py
import os


class DoubleConv(nn.Module):
    def __init__(self, In, Out):
        super().__init__()
        self.conv1 = nn.Conv1d(In, Out, 3, 1, 1)
        self.conv2 = nn.Conv1d(Out, Out, 3, 1, 1)
        self.conv3 = nn.Conv1d(Out, Out, 3, 1, 1)
        self.conv4 = nn.Conv1d(Out, Out, 3, 1, 1)
        self.conv5 = nn.Conv1d(Out, Out, 3, 1, 1)
        self.norm = nn.BatchNorm1d(Out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class Down(nn.Module):
    def __init__(self, In, Out):
        super(Down, self).__init__()
        self.pool1 = nn.MaxPool1d(2, 2)
        self.double_conv = DoubleConv(In, Out)

    def forward(self, x):
        x = self.pool1(x)
        x = self.double_conv(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.double_conv = DoubleConv(1, 32) # (batch, 1, 3200) -> (batch, 32, 3200)
        self.down1 = Down(32, 64) # (batch, 32, 3200) -> (batch, 64, 1600)
        self.down2 = Down(64, 128) # (batch, 64, 1600) -> (batch, 128, 800)
        self.down3 = Down(128, 128) # (batch, 128, 800) -> (batch, 128, 400)
        self.down4 = Down(128, 128) # (batch, 128, 400) -> (batch, 128, 200)
        self.down5 = Down(128, 128) # (batch, 128, 200) -> (batch, 128, 100)
        self.pool = nn.MaxPool1d(5, 5) # (batch, 128, 100) -> (batch, 128, 20)
        self.conv1 = nn.Conv1d(128, 128, 3, 1, 1) # (batch, 128, 20) -> (batch, 128, 20)
        self.conv2 = nn.Conv1d(128, 64, 3, 1, 1) # (batch, 128, 20) -> (batch, 64, 20)
        self.conv3 = nn.Conv1d(64, 32, 3, 1, 1) # (batch, 64, 20) -> (batch, 32, 20)
        self.conv4 = nn.Conv1d(32, 16, 3, 1, 1) # (batch, 32, 20) -> (batch, 16, 20)
        self.conv5 = nn.Conv1d(16, 12, 3, 1, 1) # (batch, 16, 20) -> (batch, 12, 20)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(32)
        self.norm4 = nn.BatchNorm1d(16)
        self.norm5 = nn.BatchNorm1d(12)
        self.conv = nn.Conv1d(12, 12, 1, 1) # (batch, 12, 20) -> (batch, 12, 20)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.conv(x)
        return x


inputs_path = 'C:/Users/Ryuusei/Desktop/VOXceleb/VoxTrain/wav_python'
targets_path = 'C:/Users/Ryuusei/Desktop/VOXceleb/VoxTrain/mfcc_matlab'
model_PATH = 'C:/Users/Ryuusei/PycharmProjects/XPRIZE/DANV-master/h5'

net = Net()
net = net.double()
net.load_state_dict(torch.load('model.pth'))
net.eval()

optimizer = optim.Adam(net.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0

with h5py.File(inputs_path + '/wav_python' + str(1) + '.h5', 'r') as inputs_data:
    a_group_key = list(inputs_data.keys())[0]
    inputs = inputs_data[a_group_key]
    inputs_s2 = inputs.shape[0]
    inputs = inputs[0]
    inputs_s = inputs.shape[0]
    print(inputs)
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (1, 1, 3200))
    inputs = torch.from_numpy(inputs)
    inputs = inputs.double()
    start = time.time()
    with h5py.File(targets_path+ '/mfcc_matlab' + str(1) + '.h5', 'r') as targets_data:
            a_group_key = list(targets_data.keys())[0]
            targets = targets_data[a_group_key]
            targets_s2 = targets.shape[0]
            targets = targets[0]
            if not np.isnan(targets).any():
                targets2 = targets
                targets_s = targets.shape[0]
                targets = np.array(targets)
                targets = np.reshape(targets, (1, 1, 12, 20))
                targets = torch.from_numpy(targets)
                targets = targets.double()
                trained = net(inputs)
                trained_s = trained.size()[0]
                trained = torch.reshape(trained, (trained_s, 1, 12, 20))
                loss = nn.L1Loss()
                output = loss(trained, targets)
                optimizer.zero_grad()
                output.backward()
                optimizer.step()
                running_loss += output.item()
                end = time.time()
                period = end - start
                print('loss: %.5f' %
                      running_loss)
                print('python : ', trained)
                print('matlab : ', targets)
                print('time/step: %.5f' % period)
