import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from image_alteration_v2 import default_alter, draw_points

# Constants
WIDTH = 240
HEIGHT = 135
CROP = int(HEIGHT * 0.259525)
CROPPED_HEIGHT = HEIGHT - CROP
LABELS_PATH = 'lane-detection-dataset/labels/'
VAL_LABELS_PATH = 'lane-detection-dataset/val_labels/'

# Data Preparation
def img_crop(img, rl, ll):
    img = img[CROP:, :, :]
    rl = np.asarray(rl)
    ll = np.asarray(ll)
    rl[:, 1] -= CROP
    ll[:, 1] -= CROP
    return img, rl, ll

def img_preprocess(img, right_lane, left_lane):
    img, rl, ll = img_crop(img, right_lane, left_lane)
    img = cv2.resize(img, (WIDTH, CROPPED_HEIGHT))
    rl = np.asarray([rl[:,0] / WIDTH, rl[:,1] / CROPPED_HEIGHT]).transpose()
    ll = np.asarray([ll[:,0] / WIDTH, ll[:,1] / CROPPED_HEIGHT]).transpose()
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img, rl, ll

# Model
model = load_model("model_1.h5")

# Display Processed Images
def display_processed_images(labels_path):
    labels_paths = sorted(os.listdir(labels_path))
    for label_file in labels_paths:
        label = pickle.load(open(os.path.join(labels_path, label_file), "rb"))
        img, rl, ll = label['img'], label['right_lane'], label['left_lane']
        img, rl, ll = img_preprocess(img, rl, ll)
        img = draw_points(img, rl, ll)
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    display_processed_images(VAL_LABELS_PATH)
