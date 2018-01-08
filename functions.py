# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:48:57 2018

@author: asd
"""

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

###############################################################################
### Helper functions

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    ''' Draw bounding boxes on an iamge '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_labeled_bboxes(img, labels, color=(0, 0, 255), thick=6):
    ''' Draw bounding boxes on image based on label image'''
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    
    img = draw_boxes(img, bboxes, color, thick)
    return img

'''
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
'''

def convert_color_from_RGB(image, color_space='RGB'):
    if color_space == 'RGB':
        feature_image = np.copy(image)
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        print('No valid color space was selected')
    return feature_image

###############################################################################
###  Feature extraction

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    ''' Extraction of HOG features and visualization '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
 
def bin_spatial(img, size=(32, 32), channel=0):
    ''' Extraction of binned color features '''
    if channel=='ALL':
        features = cv2.resize(img, size).ravel()
    else:
        features = cv2.resize(img[:,:,channel], size).ravel() 
    return features

def color_hist(img, nbins=32, bins_range=(0, 1)):
    ''' Extraction of color histogram features
    jgp:bins_range=(0,256)'''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

###############################################################################
###

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True): 
    ''' Extract all desired features of an image
    Parameters
    ----------
    <write explanations>
    Returns
    -------
    np.array
        concatenated feature vector in order spatial, color, hog
    '''
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        feature_image = convert_color_from_RGB(img, color_space)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Returns
    -------
    list of np.array
        list of feature vectors of every image
    '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        image = mpimg.imread(file)
        
        file_features =  single_img_features(image, color_space=color_space,
                                             spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat,
                                hog_feat=hog_feat)

        features.append(file_features)
    # Return list of feature vectors
    return features

###############################################################################
### Image subsampling
    

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    Creates a list of windows for a given image
    Parameters
    ----------
    img : np.array
        image
    x_start_stop : 2-list
        start and stop positions in x
    y_start_stop : 2-list
        start and stop positions in y
    xy_window : 2-tuple
        window size (x and y dimensions)
    xy_overlap : 2-tuple
        overlap fraction (for both x and y)
    Returns
    -------
    list of 4-tuple
        list of bounding windows
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    '''
    Classify an image within windows wether a car is present
    
    Parameters
    ----------
    img : np.array
        image (RGB, jpg?) on which cars should be detected
    windows : list of 4-tuples
        bounding boxes of windows which will be classified
    clf : sklearn.Classifier
        trained classifier
    scaler : sklearn.StandardScaler
        scaler of the features
    '''
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

###############################################################################

def find_cars(image, clf, X_scaler, scale=1,
              x_start_stop=[None, None], y_start_stop=[None, None],
              orient=9, pix_per_cell=8, cell_per_block=2,
              hog_channel=0,
              spatial_size=(32,32), hist_bins=32):
    '''
    Extract features using hog sub-sampling and make predictions
    Parameters
    ----------
    img: np.array
        *.jpg image in RGB color space
    clf: sklearn.Classifier
        trained classifier
    X_Scaler: sklearn.StandardScaler
    
    '''
    
    draw_img = np.copy(image)
    img = np.copy(image)
    img = img.astype(np.float32)/255
    
    # cut image
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    
    img_tosearch = img[y_start_stop[0]:y_start_stop[1],
                       x_start_stop[0]:x_start_stop[1]]

    img_color = convert_color_from_RGB(img_tosearch, 'YCrCb')
    if scale != 1:
        imshape = img_color.shape
        img_color = cv2.resize(img_color, (np.int(imshape[1]/scale),
                                      np.int(imshape[0]/scale)))

    shape_new = img_color.shape

    if hog_channel == 'ALL':
        hog = []
        for channel in range(shape_new[2]):
            hog.append(get_hog_features(img_color[:,:,channel], orient, pix_per_cell,
                                   cell_per_block, feature_vec=False)    )
    else:
        hog = get_hog_features(img_color[:,:,hog_channel], orient, pix_per_cell,
                                   cell_per_block, feature_vec=False)

    ### Resample the hog features

    # Define blocks and steps as above
    nxblocks = (shape_new[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (shape_new[0] // pix_per_cell) - cell_per_block + 1 
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    

    windows_car = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(img_color.shape[2]):
                    hog_features.extend(hog[channel][ypos:ypos+nblocks_per_window,
                                    xpos:xpos+nblocks_per_window].ravel())
                hog_features = np.array(hog_features).astype(np.float64)

            else:
                hog_features = hog[ypos:ypos+nblocks_per_window,
                                   xpos:xpos+nblocks_per_window].ravel()
            '''    
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            '''
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_color[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get spatial and color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            X = np.concatenate([spatial_features, hist_features, hog_features])
            X = np.array(X).reshape(1, -1)
            #X = np.vstack((spatial_features, hist_features, hog_features)).astype(np.float64)
            # Udacity:
            #X = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(X)    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = clf.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                startx = xbox_left
                starty = ytop_draw + y_start_stop[0]
                endx = xbox_left + win_draw
                endy = ytop_draw + win_draw + y_start_stop[0]
                windows_car.append(((startx, starty), (endx, endy)))
                
                #cv2.rectangle(draw_img, (xbox_left, ytop_draw+y_start_stop[0]),
                #              (xbox_left+win_draw,ytop_draw+win_draw+y_start_stop[0]),
                #              (0,0,255), 6) 
                
    return draw_img, windows_car

###############################################################################
### Heatmaps to reduce false positive and label detections
    
def add_heat(heatmap, bbox_list):
    ''' add bboxes to a heat map'''
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    ''' Zero out pixels below the threshold '''
    heatmap[heatmap <= threshold] = 0
    return heatmap