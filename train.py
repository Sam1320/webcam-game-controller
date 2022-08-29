import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import env
from utils.create_dataset import create_dataset


def print_accuracy(data_loader, model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            incorrect = predicted != labels
            labels_np = np.array(labels)
            predicted_np = np.array(predicted)
            images_np = np.array(images)
            outputs_np = np.array(outputs)
            print()
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


def train_model(model, criterion, optimizer, trainloader, epochs, verbose=False):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

        if epoch % 10 == 0 and verbose:
            print(f'epoch = {epoch} | loss: {running_loss / i}')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
      super(CNN2, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # Number of Linear input connections depends on output of conv2d layers
      # and therefore the input image size, so compute it.
      def conv2d_size_out(size, kernel_size=3, stride=1):
          return (size - (kernel_size - 1) - 1) // stride + 1

      convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))
      convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))
      linear_input_size = convw * convh * 64

      self.fc1 = nn.Linear(linear_input_size, 128)
      self.fc2 = nn.Linear(128, 3)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output


class DQN(nn.Module):
    def __init__(self, h=64, w=64, outputs=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw*convh*32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class NN(nn.Module):
    def __init__(self, n_hidden=128):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(64 * 64, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(x, dim=1)
        return output


def run_experiment(epochs, batch_size, save=False):
    batch_size_train = batch_size
    batch_size_test = len(x_test) - 1
    trainset = TensorDataset(x_train, y_train)
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    testset = TensorDataset(x_test, y_test)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)
    model = DQN(64, 64, 3)

    # model = NN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model=model, criterion=criterion, optimizer=optimizer, trainloader=trainloader, epochs=epochs)
    print_accuracy(trainloader, model)
    print_accuracy(testloader, model)
    if save:
        path = os.path.join(env.models_path, "cnn_v1.pt")
        print(f"saving model in {path}")
        torch.save(model.state_dict(), path)
    print("----" * 20)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = create_dataset(processed_imgs_path=env.images_processed_path, train_split=0.99)
    x_train = x_train.transpose(0, 3, 1, 2)
    x_train = torch.from_numpy(x_train.astype(np.float32))
    y_train = torch.from_numpy(y_train).type(torch.long)

    x_test = x_test.transpose(0, 3, 1, 2)
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_test = torch.from_numpy(y_test).type(torch.long)
    assert x_train.shape[1:] == (1, 64, 64)

    transform = transforms.Normalize([0.5], [0.5])
    x_train = transform(x_train)
    x_test = transform(x_test)

    batch_size = len(x_train)-1
    batch_sizes = [32]

    for batch_size in batch_sizes:
        print(f"BATCH SIZE: {batch_size}")
        for i in range(1):
            run_experiment(epochs=20, batch_size=batch_size, save=True)

