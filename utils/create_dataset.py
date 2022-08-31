import os
import cv2
import math
import numpy as np


def load_images(processed_imgs_path):
    folders = ["left", "right", "wait"]
    n_files = sum([len(os.listdir(os.path.join(processed_imgs_path, folder))) for folder in folders])
    x = np.zeros((n_files, 64, 64, 1))
    y = np.zeros(n_files)
    file_idx = 0
    for label, folder in enumerate(folders):
        folder_path = os.path.join(processed_imgs_path, folder)
        for file in os.listdir(folder_path):
            filename = os.path.join(folder_path, file)
            x_tmp = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            x_tmp = np.expand_dims(x_tmp, axis=2)
            x[file_idx, :, :] = x_tmp
            y[file_idx] = label
            file_idx += 1
    return x, y


def create_dataset(processed_imgs_path, train_split=0.8):
    # n_folds = 1
    # for fold_n in range(n_folds):
    x, y = load_images(processed_imgs_path=processed_imgs_path)
    n_train_samples = math.floor(len(x) * train_split)
    indexes = np.array(range(len(x)))
    np.random.shuffle(indexes)

    train_idx = indexes[:n_train_samples]
    test_idx = indexes[n_train_samples:]

    x_train = x[train_idx, :, :]
    y_train = y[train_idx]

    x_test = x[test_idx, :, :]
    y_test = y[test_idx]

    assert len(x_test)+len(x_train) == len(x)
    return x_train, y_train, x_test, y_test



