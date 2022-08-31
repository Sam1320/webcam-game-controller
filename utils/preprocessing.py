import cv2
import matplotlib.pyplot as plt
import os

import env


def crop(im, w_crop=.25, h_crop=.2):
    h, w = im.shape
    h_margin, w_margin = int(h*h_crop), int(w*w_crop)
    return im[h_margin:-h_margin, w_margin:-w_margin]


def resize(im, shape=(64, 64)):
    width = int(im.shape[1] * shape[1]/ im.shape[1])
    height = int(im.shape[0] * shape[0]/ im.shape[0])
    return cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)


def black_white(im):
    return cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 63, -30)


def plot_images(im_list, titles=None):
    f, axarr = plt.subplots(1, len(im_list), figsize=(10,10))
    for i, ax, im in zip(range(len(im_list)), axarr, im_list):
        ax.imshow(im, cmap=plt.cm.gray)
        ax.axis("off")
        if titles:
            ax.title.set_text(titles[i])
    plt.show()


def process_image(im):
    return black_white(resize(crop(im)))


def add_test_to_train():
    """Rename and move the files in *_test folders to the train folders."""
    train_path = f"{env.images_processed_path}"
    test_path = train_path + "_test"
    folder_names = list(os.listdir(train_path))
    n_files = {folder_name: len(os.listdir(os.path.join(train_path, folder_name))) for folder_name in folder_names}
    answer = input("You are about to rename and move all the files in the test folders. continue? y/n:")
    while answer not in {'y', 'n'}:
        answer = input(f"{answer} is not a valid option. please select 'y' or 'n':")
    if answer == 'y':
        print(f"Renaming and moving files from {test_path} to {train_path}")
        for folder in os.listdir(test_path):
            for i, file in enumerate(os.listdir(os.path.join(test_path, folder))):
                filename = os.path.join(test_path, folder, file)
                new_name = os.path.join(train_path, folder, f"{folder}_{i+n_files[folder]}.jpg")
                os.rename(filename, new_name)
    else:
        print("Aborted.")


def process_frames(src_folder, dst_folder):
    prefix = os.path.split(src_folder)[1]
    for i, filename in enumerate(os.listdir(src_folder)):
        im_path = os.path.join(src_folder, filename)
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        im_processed = process_image(im)
        new_filename = f"{prefix}_{i}.jpg"
        cv2.imwrite(os.path.join(dst_folder, new_filename), im_processed)