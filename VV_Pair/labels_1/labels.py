import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import pickle
import time

# Video file path
video_file_path = 'valid.mp4'

# Set up the OpenCV window
cv2.namedWindow("left: green --- right: blue", cv2.WINDOW_KEEPRATIO)
video_capture = cv2.VideoCapture(video_file_path)

def set_lane_points(event, x, y, flags, params):
    img = params['image']
    click_count = params['click_count']
    right_lane_points = params['right_lane_points']
    left_lane_points = params['left_lane_points']
    top_boundary = params['top_boundary']
    bottom_boundary = params['bottom_boundary']

    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count == 0:
            top_boundary = y
            y_positions = np.geomspace(top_boundary, bottom_boundary, 6).astype(int)
            for y_pos in y_positions:
                left_lane_points.append([0, y_pos])
                right_lane_points.append([0, y_pos])
                cv2.line(img, (0, y_pos), (img.shape[1], y_pos), color=(0, 0, 255), thickness=2)
        elif 1 <= click_count <= 6:
            left_lane_points[click_count - 1][0] = x
            cv2.circle(img, (x, left_lane_points[click_count - 1][1]), radius=3, thickness=6, color=(0, 255, 0))
        elif 7 <= click_count <= 12:
            right_lane_points[click_count - 7][0] = x
            cv2.circle(img, (x, right_lane_points[click_count - 7][1]), radius=3, thickness=3, color=(255, 0, 0))
        params['lane_image'] = copy.deepcopy(img)
        click_count += 1
        if click_count == 13:
            params['done'] = True

    if event == cv2.EVENT_MOUSEMOVE and click_count > 0:
        img = copy.deepcopy(params['lane_image'])
        if 1 <= click_count <= 6:
            cv2.circle(img, (x, left_lane_points[click_count - 1][1]), radius=3, thickness=3, color=(0, 255, 0))
        elif 7 <= click_count <= 12:
            cv2.circle(img, (x, right_lane_points[click_count - 7][1]), radius=3, thickness=3, color=(255, 0, 0))

    params['image'] = img
    params['click_count'] = click_count
    params['top_boundary'] = top_boundary
    params['bottom_boundary'] = bottom_boundary
    params['right_lane_points'] = right_lane_points
    params['left_lane_points'] = left_lane_points

if __name__ == "__main__":
    lane_params = {
        'image': None,
        'click_count': 0,
        'right_lane_points': [],
        'left_lane_points': [],
        'top_boundary': 0,
        'bottom_boundary': 720,
        'lane_image': None,
        'done': False
    }

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        lane_params['image'] = frame
        cv2.setMouseCallback("left: green --- right: blue", set_lane_points, lane_params)

        while not lane_params['done']:
            cv2.imshow("left: green --- right: blue", lane_params['image'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if lane_params['done']:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    with open('lane_labels.pkl', 'wb') as file:
        pickle.dump({
            'right_lane_points': lane_params['right_lane_points'],
            'left_lane_points': lane_params['left_lane_points']
        }, file)
