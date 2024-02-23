import os

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

import env
from utils import preprocessing

# todo: needs to be integrated in model pipeline
transform = transforms.Normalize([0.5], [0.5])


def test_examples(model):
    train_images = ['images/processed/left/left_12.jpg',
                    'images/processed/right/right_12.jpg',
                    'images/processed/wait/wait_12.jpg']

    realtime_images = ['/home/sam/Code/webcam_game_controller/left.jpg',
                       '/home/sam/Code/webcam_game_controller/right.jpg',
                       '/home/sam/Code/webcam_game_controller/wait.jpg']

    test_images = ["images/processed_test/left/left_0.jpg",
                   "images/processed_test/right/right_17.jpg",
                   "images/processed_test/wait/wait_190.jpg"]
    fig, axarr = plt.subplots(3, 3)
    for ax, filename in zip(axarr[0], train_images):
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        prediction = model_predict(model, im)
        ax.imshow(im)
        ax.title.set_text(f"(train) {prediction}")
    for ax, filename in zip(axarr[1], test_images):
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        prediction = model_predict(model, im)
        ax.imshow(im)
        ax.title.set_text(f"(test) {prediction}")
    for ax, filename in zip(axarr[2], realtime_images):
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        prediction = model_predict(model, im)
        ax.imshow(im)
        ax.title.set_text(f"(real-time) {prediction}")
    plt.show()


def plot_n_images_with_prediction(rows, cols, folder, model):
    n = rows*cols
    files = list(os.listdir(folder))[:n]
    fig, axarr = plt.subplots(rows, cols, figsize=(20,20))
    fig.suptitle(f"folder = {folder}")
    for row in range(rows):
        for col in range(cols):
            filename = files.pop()
            im = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            assert im is not None
            prediction = model_predict(model, im)
            axarr[row][col].imshow(im)
            axarr[row][col].title.set_text({prediction})
    plt.show()


def take_and_store_img(filepath):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    im = np.array(frame)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_processed = preprocessing.process_image(im)
    cv2.imwrite(filepath, im_processed)
    cap.release()
    cv2.destroyAllWindows()


def model_predict(model, im, raw=False):
    if raw:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = preprocessing.process_image(im, )
    actions = ['left', 'right', 'wait']
    im_tensor = torch.from_numpy(im.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    im_normalized = transform(im_tensor)
    with torch.no_grad():
        outputs = model(im_normalized)
        _, predicted = torch.max(outputs.data, 1)
        if raw:
            return im, actions[predicted]
        return actions[predicted]


def record_samples():
    for i in ['left', 'right', 'wait']:
        input(f"press enter to take image of action = {i}:")
        filepath = f"{env.base_path}/{i}.jpg"
        take_and_store_img(filepath)


def realtime_labelling(model):
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    model.eval()
    while cap.isOpened():
        ret, frame = cap.read()
        im = np.array(frame)
        im_processed, prediction = model_predict(model, im, raw=True)
        im_processed = cv2.resize(im_processed, (im.shape[1], im.shape[0]))
        cv2.putText(im, prediction, (250, 250), font, 1, (0, 255, 0), 3)
        cv2.imshow('original', im)
        cv2.imshow('processed', im_processed)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


def run_system(model=None):
    action_mappings = {'left': 0, 'right': 2, 'wait': 1}
    env = gym.make('MountainCar-v0', render_mode='human')
    env.reset()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    model.eval()
    while True and cap.isOpened():
        ret, frame = cap.read()
        im = np.array(frame)
        im_processed, prediction = model_predict(model, im, raw=True)
        im_processed = cv2.resize(im_processed, (im.shape[1], im.shape[0]))
        text_size, _ = cv2.getTextSize(prediction, font, 5, 3)
        text_x = (im.shape[1] - text_size[0]) // 2
        text_y = (im.shape[0] + text_size[1]) // 2
        cv2.putText(im, prediction, (text_x, text_y), font, 5, (0, 255, 0), 3)
        cv2.imshow('original', im)
        cv2.imshow('processed', im_processed)

        action = action_mappings[prediction]
        state = env.step(action)[0]
        position = state[0]
        done = position >= 0.5
        if done:
            env.reset()
        env.render()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break