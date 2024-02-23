import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNv1(nn.Module):
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#:~:text=class%20DQN(nn.Module)%3A
    def __init__(self, input_height=64, input_width=64, outputs=3, n_kernels=16, kernel_size=5, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  n_kernels, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.conv2 = nn.Conv2d(n_kernels, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
        self.bn3 = nn.BatchNorm2d(32)

        # compute output of convolutional layers
        def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height)))
        linear_input_size = conv_width*conv_height*32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch during optimization
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class CNNv2(nn.Module):
    def __init__(self, input_height=64, input_width=64, outputs=3, n_kernels=3, kernel_size=3, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 4, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=kernel_size, stride=stride)

        # compute output of convolutional layers
        def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height)))
        linear_input_size = conv_width * conv_height * 4
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch during optimization
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))


class NNv1(nn.Module):
    def __init__(self, n_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(x, dim=1)
        return output


class NNv2(nn.Module):
    def __init__(self, n_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = F.log_softmax(x, dim=1)
        return output