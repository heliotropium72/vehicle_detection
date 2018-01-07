# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Term 1, Project 5: Vehicle Detection
### Keywords: Computer Vision, Machine Learning, ...
Note: This project will be (eventually) united with project 4: [Advanced Lane Finding](https://github.com/heliotropium72/lane_detection.git) and is (already now) largely build on this project.

### Work in Progress 7/1/2018

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

In this project, vehicles are detected on a highway course based on computer vision and machine learning.
Checkout the [project rubric](https://review.udacity.com/#!/rubrics/513/view) for more details.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

  

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!

---
## Image Classifier (Vehicle/ No-vehicle)
As a first step an image classifier, which is able to predict whether an image snipplet contains a vehicle, is trained.

### Data set
Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and
[non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier. 
These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/),
and examples extracted from the project video itself. The dataset could be further augmented with the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) .

The data set contains 8792 images of vehicles and 8968 images of non-vehicles. The images are in RGB color space, 64x64 pixel large and in the png format.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### Features
Before the feature extraction, the image are scaled to 0 to 1 if they are jpg files.
Then they are converted to the YCrCb color space, where Y (luma) is representing the black-and-white image and the Cr and Cb channels represent "chroma" or color.
The Y channel can be effectively used for color-independent gradients (here hog).

#### Spatially binned colors
--> overall spatial orientation
#### Color histograms
--> overall color

#### Histogram of Oriented Gradients (HOG)
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image2]

#### Normalisation
The features were then combined to a single feature vector which was normalised using `sklearn.StandardScaler`

### Support vector classifier (SVC)
The data was split into a training and test data set. Then a svc was trained.
Training took about 10s and the accuracy on the test data set was > 97%.
The trained classifier is used in the following for the vehicle tracking on a video stream.

---
## Vehicle recognition

### Sliding window 
I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

### Classification
Every selected image snippet was then classified using the above trained SVC. In order to avoid false positive (detected cars in an empty snippet), a heatmap was created.
All pixel within a positive (true and false) window were added with 1 point to a heatmap.
That was done for all windows in the current image and the previous two images. Then a threshold was applied to remove false positives, which were usually not consistent over many windows and images.
Using `scipy.ndimage.measurements.label()` new bounding boxes (one for each car) were drawn on the image.

---

## Video Implementation

Here's a [link to my video result](./Videos/video1.mp4)

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

## Discussion

#### Problems
- Separation of cars: When two vehicles are close to each other, they will be detected as a single vehicle
- False positive: The pipeline does not reject all outliers

#### Improvements
- Image classification: The image classification could made more robust by performing a grid search tuning through a variety of hyper-parameters and also different classifier (e.g decision tree)
- Image classification: The data set could be augmented to improve prediction accuracy even further
- Sliding Window: In the current implementation the image is only separated into a part which is searched and a ignored part. In the whole searched part, the same scales are applied for the windows. An algorithm which decreased the scaled
towards the image borders would be beneficial in the sense of efficiency and robustness (e.g. a large search window in the upper left corner does not make sense).
- Combination with lane detection: In theory, the vehicle detection pipeline could be easily combined with the lane detection pipeline to give a more complete result. Also the information about the lane could be beneficial in order to check
the detection of vehicles: Are they driving on a lane? In which direction are they driving?