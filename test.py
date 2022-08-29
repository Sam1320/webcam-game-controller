import os
from train import DQN, print_accuracy
from utils.create_dataset import create_dataset
import env
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import cv2

model = DQN()
model.load_state_dict(torch.load(os.path.join(env.models_path, 'cnn_v1.pt')))
model.eval()

test_imgs_path = f"{env.images_processed_path}_test"
_, _, x_test, y_test = create_dataset(train_split=0, processed_imgs_path=test_imgs_path)
x_test = x_test.transpose(0, 3, 1, 2)
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.long)
transform = transforms.Normalize([0.5], [0.5])
x_test = transform(x_test)

# batch_size = len(x_test) - 1
batch_size = 1
testset = TensorDataset(x_test, y_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

print_accuracy(testloader, model)
print()
