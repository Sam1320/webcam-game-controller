import os.path

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from models.models import *
import numpy as np
from tqdm import tqdm

import env
from utils.create_dataset import create_dataset


def calculate_accuracy(data_loader, model, verbose=False):
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#:~:text=the%20whole%20dataset.-,correct%20%3D%200,-total%20%3D%200
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
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#:~:text=network%20and%20optimize.-,for%20epoch,-in%20range(
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
            running_loss += loss.item()

        if epoch % 2 == 0 and verbose:
            print(f'epoch = {epoch} | loss: {running_loss / i}')


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
    model_names = {'nnv1': [2, 4, 8, 16, 32, 64, 128], 'nnv2': [2, 4, 8, 16, 32, 64, 128],
                   'cnnv1': [4, 8, 16], 'cnnv2': [3]}
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
            else:
                raise ValueError(f"{model_name} is not a valid model name.")
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
