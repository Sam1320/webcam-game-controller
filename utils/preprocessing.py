import cv2
import matplotlib.pyplot as plt
import os


def crop(im, w_crop=.25, h_crop=.2):
    h, w = im.shape
    h_margin, w_margin = int(h*h_crop), int(w*w_crop)
    return im[h_margin:-h_margin, w_margin:-w_margin]


def resize(im, shape=(64, 64)):
    width = int(im.shape[1] * shape[1]/ im.shape[1])
    height = int(im.shape[0] * shape[0]/ im.shape[0])
    return cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)


def black_white(im):
    return cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 63, -20)


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


def process_frames(src_folder, dst_folder):
    prefix = os.path.split(src_folder)[1]
    for i, filename in enumerate(os.listdir(src_folder)):
        im_path = os.path.join(src_folder, filename)
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        im_processed = process_image(im)
        new_filename = f"{prefix}_{i}.jpg"
        cv2.imwrite(os.path.join(dst_folder, new_filename), im_processed)


