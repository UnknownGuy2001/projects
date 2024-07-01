import os
import cv2
import copy
import numpy as np
from imgaug import augmenters as iaa
import random
import noise
from scipy import misc
from sklearn import preprocessing
from PIL import Image, ImageFilter
import time

LABELS_PATH = 'labels/'
LABELS_PATHS = os.listdir(LABELS_PATH)

def draw_points(img, right_lane, left_lane):
    for point in right_lane:
        cv2.circle(img, (point[0], point[1]), radius=3, thickness=3, color=(255, 0, 0))
    for point in left_lane:
        cv2.circle(img, (point[0], point[1]), radius=3, thickness=3, color=(0, 255, 0))
    return img

def draw_pixels(src_img, lane):
    img = np.zeros_like(src_img)
    for point in lane:
        cv2.circle(img, (point[0], point[1]), radius=1, thickness=5, color=(0, 255, 0))
    return img

def blobs_to_lane(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 3
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 500
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)
    points = sorted([list(map(int, kp.pt)) for kp in keypoints], key=lambda x: x[1])
    return points

def crop(img, width, height, center=True):
    if center:
        d_x = (img.shape[1] - width) // 2
        d_y = (img.shape[0] - height) // 2
        cropped_img = img[d_y:img.shape[0]-d_y, d_x:img.shape[1]-d_x]
    else:
        d_x = img.shape[1] - width
        d_y = img.shape[0] - height
        cropped_img = img[d_y:img.shape[0], 0:img.shape[1]-d_x//3]
    return cropped_img

def lens_distort(img, right_lane, left_lane, amount=0.15):
    distortion = iaa.PiecewiseAffine(scale=(0.01, amount))
    img = distortion.augment_image(img)
    right_lane = distortion.augment_keypoints([right_lane])[0].keypoints
    left_lane = distortion.augment_keypoints([left_lane])[0].keypoints
    return img, right_lane, left_lane

def default_alter(img, right_lane, left_lane):
    img, right_lane, left_lane = lens_distort(img, right_lane, left_lane)
    return img, right_lane, left_lane

def save_image(img, file_path):
    cv2.imwrite(file_path, img)

def process_and_save_images():
    for label_file in LABELS_PATHS:
        label = pickle.load(open(os.path.join(LABELS_PATH, label_file), "rb"))
        img, right_lane, left_lane = label['img'], label['right_lane'], label['left_lane']
        img, right_lane, left_lane = default_alter(img, right_lane, left_lane)
        img = draw_points(img, right_lane, left_lane)
        save_image(img, os.path.join('processed', label_file))

if __name__ == "__main__":
    if not os.path.exists('processed'):
        os.makedirs('processed')
    process_and_save_images()
