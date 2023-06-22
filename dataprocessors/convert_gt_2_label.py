import numpy as np
import cv2 as cv
import glob
import os

# data_road is kitti data structure
dir_path = '../data/data_road/training/gt_image_2/um_lane*.png'
save_dir_path = '../data/data_road/training/label_lane_image_2'

if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

files = np.array(sorted(glob.glob(dir_path)))

print(files)
for image in files:
    image_name = image.split('/')[-1]
    image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    #cv.imshow('原图', image)
    # THRESH = 50
    # ret, binary = cv.threshold(image, THRESH, 255, cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
    #cv.imshow('binary', binary)
    # Apply Canny edge detection to find edges
    edges = cv.Canny(image, 50, 100)
    # Save to local
    cv.imwrite(save_dir_path + '/' + image_name, edges)
    #cv.waitKey(0)


# data_road is kitti data structure
dir_path = '../data/data_road/training/gt_image_2/um_road*.png'
save_dir_path = '../data/data_road/training/label_road_image_2'

if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

files = np.array(sorted(glob.glob(dir_path)))

print(files)
for image in files:
    image_name = image.split('/')[-1]
    image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    #cv.imshow('原图', image)
    THRESH = 50
    ret, binary = cv.threshold(image, THRESH, 255, cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
    #cv.imshow('binary', binary)
    # Save to local
    cv.imwrite(save_dir_path + '/' + image_name, binary)
    #cv.waitKey(0)