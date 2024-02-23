import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from train import CNNv1, CNNv2, calculate_accuracy

import env
from utils.create_dataset import create_dataset

test_imgs_path = f"{env.images_processed_path}_test"
_, _, x_test, y_test = create_dataset(train_split=0, processed_imgs_path=test_imgs_path)
x_test = x_test.transpose(0, 3, 1, 2)
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.long)
transform = transforms.Normalize([0.5], [0.5])
x_test = transform(x_test)

batch_size = 1
testset = TensorDataset(x_test, y_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

models = {'cnnv1_4.pt': 4, 'cnnv1_8.pt': 8, 'cnnv1_16.pt': 16,  'cnnv2_3.pt': 3}
for model_name in models:
    arch = model_name.split("_")[0]
    nodes = models[model_name]
    if arch == 'cnnv1':
        model = CNNv1(n_kernels=nodes)
    elif arch == 'cnnv2':
        model = CNNv2()
    else:
        raise ValueError(f"{model_name} is not a valid model name.")
    model.load_state_dict(torch.load(os.path.join(env.models_path, model_name)))
    model.eval()
    calculate_accuracy(testloader, model, verbose=True)
