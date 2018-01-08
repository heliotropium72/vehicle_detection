# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:36:26 2018

@author: asd
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
import copy
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from scipy.ndimage.measurements import label


from moviepy.editor import VideoFileClip

'''
#importing own functions
try:
    DIST = dist.mtx
    MTX = dist.mtx
except:
    import undistort as dist
    DIST = dist.mtx
    MTX = dist.mtx
'''    
    
from functions import *   
    
###############################################################################
#
# Parameters to tune for classifier
#
#
reduced_sample_size = False

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 650] # Min and max in y to search in slide_window()
x_start_stop = [None, None]

###############################################################################
# Train a classifier

### 1) Load the data sets
directory = 'C:/Users/asd/Documents/5_CourseWork/sdc'
non_car_folder = os.path.join(directory, 'vehicle_detection_data', 'non-vehicles')
car_folder = os.path.join(directory, 'vehicle_detection_data', 'vehicles')

cars = glob.glob(car_folder + '/**/*.png', recursive=True)
notcars = glob.glob(non_car_folder + '/**/*.png', recursive=True)

if reduced_sample_size:
    sample_size = 2000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]  

### 2) Extract features according to above parameters
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

### 3) Combine and normalize features
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

### 4) Create a labels vector based on features lists
y = np.hstack((np.ones(len(car_features)), 
              np.zeros(len(notcar_features))))

### 5) Split and shuffle data
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

### 6) Train classifier
# Use a linear SVC (support vector classifier)
#svc = LinearSVC()
# Train the SVC
t1 = time.time()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
#svc.fit(X_train, y_train)
svc = clf
t2 = time.time()
print('SVC Classifier')
print('------------------')
print('Training time {:.2f} s'.format((t2-t1)))
print('Hyperparameters : ', clf.best_params_) # rbf, 10 ... 99.6%

# Check classifier
print('Using hog with:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
print('Test Accuracy : ', svc.score(X_test, y_test))

###############################################################################
# Use the classifier on an image

folder_test = 'test_images'
image_files = glob.glob(os.path.join(folder_test,'*.jpg'))
images = []

for fname in image_files:
    images.append(mpimg.imread(os.path.join(fname)))

image = images[7]
image = image.astype(np.float32)/255 # Difference between png and jpg
draw_image = np.copy(image)

'''
t1 = time.time()
### 1) Separate the image into sliding windows
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))

### 2) Classify each subimage
hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       
t2 = time.time()
print('Prediction time {:.2f} s'.format((t2-t1)))

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6) 
plt.figure()
plt.imshow(window_img)
'''
### 1+2) Alternative: Calculate hog for whole image and resample
t1 = time.time()
hot_windows = []
detections = []
for scale in [1.6, 1.8, 2]:    
    img_d, win_d = find_cars(images[7], svc, X_scaler, scale=scale,
                  x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                  orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                  spatial_size=spatial_size, hist_bins=hist_bins)
    hot_windows.extend(win_d)
    detections.append(len(win_d))
t2 = time.time()
print('Prediction time {:.2f} s'.format((t2-t1)))


img_detected = draw_boxes(draw_image, hot_windows[:detections[0]], color=(0,1,0))
img_detected = draw_boxes(img_detected,
                          hot_windows[detections[0]:detections[0]+detections[1]], color=(1,0,0))
img_detected = draw_boxes(img_detected,
                          hot_windows[detections[0]+detections[1]:], color=(0,0,1))
plt.figure()
plt.title('Detected windows')
plt.imshow(img_detected)
            
### 3) Create heatmap
heat = np.zeros_like(image[:,:,0]).astype(np.float)
heat = add_heat(heat, hot_windows)
heat = apply_threshold(heat, 4)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()


### 4) Create heatmap over k images
### 5) Return bounding boxes of detected cars


###############################################################################
# Define a single function that can extract features using hog sub-sampling and make predictions


###############################################################################
# Apply the classifier to a videostream 

def pipeline(image, scales=[1, 1.5, 2], heat_threshold=1):
    '''
    image : np.array
        Image from video stream (RGB & *.jpg)
    scales : list of float
        scale refers to 64x64 search window
    heat_threshold : int
        threshold for outlier rejection on heatmap of a single image
    '''
    # At the moment inside the find_cars function    
    #image = image.astype(np.float32)/255 # Difference between png and jpg    
    
    hot_windows = []
    for scale in scales:    
        img_d, win_d = find_cars(image, svc, X_scaler, scale=scale,
                      x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                      orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                      spatial_size=spatial_size, hist_bins=hist_bins)
        hot_windows.extend(win_d)
    img_detected = draw_boxes(np.copy(image), hot_windows, color=(0,255,0))    
            
    ### 3) Create heatmap
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, heat_threshold)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(img_detected, labels, color=(255,0,0))

    return draw_img, heat

shape = images[0].shape
empty = np.zeros((shape[0], shape[1])).astype(np.float)   
prev = [empty, empty, empty, empty, empty]#, None, None, None]
def detect_cars(image):
    ''' input image between 0 and 255'''
    img, heat = pipeline(image, scales=[1.5, 1.75, 2.0], heat_threshold=1)

    prev.append(heat)
    prev.pop(0)
    heat_summed = np.array(prev).sum(axis=0)
    heat_summed = apply_threshold(heat_summed, threshold=12)
    heatmap = np.clip(heat_summed, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels, color=(255,0,0))

    return draw_img

#import sys
#sys.exit()

video0_output = 'Videos/video0_detected.mp4'
clip0 = VideoFileClip("../lane_detection_video/video0.mp4")
video0 = clip0.fl_image(detect_cars)
video0.write_videofile(video0_output, audio=False)

video1_output = 'Videos/video1_detected.mp4'
clip1 = VideoFileClip("../lane_detection_video/video1.mp4")
video1 = clip1.fl_image(detect_cars)
video1.write_videofile(video1_output, audio=False)

video2_output = 'Videos/video2_detected.mp4'
clip2 = VideoFileClip("../lane_detection_video/video2.mp4").subclip(0,1)
video2 = clip2.fl_image(detect_cars)
video2.write_videofile(video2_output, audio=False)