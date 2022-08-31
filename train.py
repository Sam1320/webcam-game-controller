import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from tqdm import tqdm

import env
from utils.create_dataset import create_dataset


def calculate_accuracy(data_loader, model, verbose=False):
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
            # todo: analyze per class accuracy
            incorrect = predicted != labels
    accuracy = 100 * correct // total
    if verbose:
        print(f'Accuracy of the network on the {total} images: {accuracy} %')
    return accuracy


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

        if epoch % 2 == 0 and verbose:
            print(f'epoch = {epoch} | loss: {running_loss / i}')


class CNNv1(nn.Module):
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
        super(NNv1, self).__init__()
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
        super(NNv2, self).__init__()
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


def run_experiment(model, epochs, batch_size, save=False, verbose=False, model_name=None):
    batch_size_train = batch_size
    batch_size_test = len(x_test) - 1
    trainset = TensorDataset(x_train, y_train)
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    testset = TensorDataset(x_test, y_test)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model=model, criterion=criterion, optimizer=optimizer, trainloader=trainloader, epochs=epochs,
                verbose=verbose)
    train_acc = calculate_accuracy(trainloader, model)
    test_acc = calculate_accuracy(testloader, model)
    if save:
        n_models = len(os.listdir(env.models_path))
        path = os.path.join(env.models_path, f"{model_name}.pt")
        print(f"saving model in {path}")
        torch.save(model.state_dict(), path)
    return train_acc, test_acc


def count_parameters(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = create_dataset(processed_imgs_path=env.images_processed_path, train_split=0.8)
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

    n_exp = 100
    n_epochs = 10
    batch_size = 32
    # model_names = {'cnnv1': [4, 8, 16]} #,
    model_names =  {'cnnv2': [4]}
    # {'nnv1': [2, 4, 8, 16, 32, 64, 128], 'nnv2': [2, 4, 8, 16, 32, 64, 128],
    for model_name in model_names:
        for nodes in model_names[model_name]:
            if model_name == 'nnv1':
                model = NNv1(n_hidden=nodes)

            elif model_name == 'nnv2':
                model = NNv2(n_hidden=nodes)
            elif model_name == 'cnnv1':
                model = CNNv1(n_kernels=nodes)
            elif model_name == 'cnnv2':
                model = CNNv2()
            n_params = count_parameters(model)
            print(f"model {model_name}_{nodes} has {n_params} trainable parameters")



    print(f"Training samples = {len(x_train)}")
    print(f"Validation samples = {len(x_test)}")

    for model_name in model_names:
        for n_nodes in model_names[model_name]:
            train_accs = []
            test_accs = []
            for i in tqdm(range(n_exp), position=0, leave=True):
                # models need to be reset before each new experiment
                if model_name == 'nnv1':
                    model = NNv1(n_hidden=n_nodes)
                elif model_name == 'nnv2':
                    model = NNv2(n_hidden=n_nodes)
                elif model_name == 'cnnv1':
                    model = CNNv1(n_kernels=n_nodes)
                else:
                    model = CNNv2()

                train_acc, test_acc = run_experiment(model=model, epochs=n_epochs, batch_size=batch_size, save=False,
                                                     verbose=False, model_name=model_name)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
            print()
            print(f"model {model_name} | n_nodes/kernels {n_nodes} | n_experiments {n_exp} | n_epochs = {n_epochs}.")
            print(f"Train accs: {train_accs}")
            print(f"Test accs: {test_accs}")
            print(f"Average train accuracy: {sum(train_accs) / len(train_accs)}")
            print(f"Average test accuracy: {sum(test_accs) / len(test_accs)}")
