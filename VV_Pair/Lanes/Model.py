import os
import pickle
import random
import numpy as np
import cv2
import multiprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from image_alteration_v2 import default_alter, draw_points


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

def batch_generator(labels_path, batch_size, is_training):
    labels_paths = os.listdir(labels_path)
    while True:
        batch_img = []
        batch_points = []
        for i in range(batch_size):
            label = pickle.load(open(os.path.join(labels_path, labels_paths[i]), "rb"))
            img, rl, ll = label['img'], label['right_lane'], label['left_lane']
            if is_training:
                img, rl, ll = default_alter(img, rl, ll)
            img, rl, ll = img_preprocess(img, rl, ll)
            batch_img.append(img)
            batch_points.append([rl, ll])
        yield np.array(batch_img), np.array(batch_points)

# Model Definition
def build_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu', input_shape=(CROPPED_HEIGHT, WIDTH, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(4))  # Output layer for lane points
    model.compile(Adam(lr=1e-4), loss='mean_squared_error')
    return model

if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 10

    # Load datasets
    train_generator = batch_generator(LABELS_PATH, BATCH_SIZE, True)
    val_generator = batch_generator(VAL_LABELS_PATH, BATCH_SIZE, False)

    # Model training
    model = build_model()
    model.fit(train_generator,
              steps_per_epoch=200,
              epochs=EPOCHS,
              validation_data=val_generator,
              validation_steps=200)

 
    model.save('lane_detection_model.h5')
