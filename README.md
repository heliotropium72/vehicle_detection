# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Term 1, Project 5: Vehicle Detection
### Keywords: Image Classification, Computer Vision, Machine Learning
Note: This project will be (eventually) united with project 4: [Advanced Lane Finding](https://github.com/heliotropium72/lane_detection.git) and is (already now) largely build on this project.

[//]: # (Image References)

[image1]: ./Figures/data_example.png "Example data"
[image11]: ./Figures/YCrCb.png "Color space YCrCb"
[image12]: ./Figures/YCrCb_16x16.png "Spatially binned"
[image13]: ./Figures/YCrCb_hist.png "Color histogram"
[imageHOG]: ./Figures/YCrCb_hog.png "HOG features"
[image2]: ./Figures/Detections.png "Detection"
[image3]: ./Figures/Heatmap.png "Heatmap"

---

In this project, vehicles are detected on a highway course based on computer vision and machine learning.
The project is separated in two parts. First, an image classifier is trained recognizing vehicles in a 64x64 image. Then, the classifier is used to detect and track vehicles on a video stream.
Checkout the [project rubric](https://review.udacity.com/#!/rubrics/513/view) for more details.

---

## 1. Image Classifier (Vehicle/ No-vehicle)
As a first step an image classifier, which is able to predict whether an image snippet contains a vehicle, is trained.

### Data set
The classifier is trained based on this labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and
[non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples. These example images come from a combination of the
[GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.
The dataset could be further augmented with the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) .

Short summary of the data set

|Info| Value|
|:---:|:---:|
| vehicle images | 8792 |
| non-vehicle images | 8968 |
| image size | 64x64 |
| file format | png |
| color space* | RGB |
| scale* | 0 to 1 |

*when read-in with `matplotlib`


Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### Features
Before the feature extraction, the image are scaled to 0 to 1 if they are jpg files.
Then they are converted to the YCrCb color space, where Y (luma) is representing the black-and-white image and the Cr and Cb channels represent "chroma" or color.
The Y channel can be effectively used for color-independent gradients (here hog).

![alt text][image11]

#### Spatially binned colors `bin_spatial()`
The images are spatially binned using `cv2.resize(<image>, <new_size>).ravel()` to conserve the shape of the vehicles while reducing the dimensionality.

![alt text][image12]

#### Histogram of color channels `color_hist()`
The channels of the YCrCb image are converted independently into histograms using `np.histogram(<image_channel>)`. In this way the predominant color is included in the feature vector.
That is beneficial as vehicles can have strong and homogeneous color/hue which does not appear otherwise. E.g. yellow car, black tires, ...

![alt text][image13]

#### Histogram of Oriented Gradients (HOG) `get_hog_features()`
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][imageHOG]

#### Final choice of parameters
The full feature vector is extracted automatically for a single image `single_img_features(<image>)` or a list of images `extract_features(<image_list>). The user can set all parameters as options.
Also the different features (spatial, color and HOG) can be (de-)activated using a flag. Empirically, the following parameters showed the best result for the project

| Parameter 		| Value |
|:-----------------:|:-----:|
| Color space 		| YCrCb |
| Spatial feature 	| yes |
| Spatial size 		| 16x16 |
| Color feature 	| yes |
| Histogram bins 	| 16 |
| HOG feature 		| yes |
| HOG orientations 	| 9 |
| HOG pixels per cell | 8 |
| HOG cells per block | 2 |
| HOG image channel | 0 (Y) |

#### Normalisation
The features were then combined to a single feature vector which was normalised using `sklearn.StandardScaler`

### Support vector classifier (SVC)
The data was split into a training and test data set. Then a svc was trained.
A grid search using kernel={"linear", "rbf"} and C={1, 10} showed the best result with an accuracy of 99.6%.
Due to the long training time (>30min), the hyperparameters were tuned only a single time and then continuously applied.
The training of a single, linear svm took about 10s and the accuracy on the test data set was > 97%.
The trained classifier is used in the following for the vehicle tracking on a video stream.

---

## 2. Vehicle recognition
In order to detect vehicles in an image, the image is sub-sampled into smaller, overlapping snippets which are classified. A reach which was classified as vehicle by sufficiently many snippets will be marked as vehicle.
Since the recognition should be applied on a video stream, a focus on efficiency is made.

### Sliding window 
First, the image was cut to the area in which vehicles are actually present by using the `x_start_stop` and `y_start_stop` parameters. In this way the sky was cut away leading to a faster processing.
Then, the HOG features are calculated for the whole image, which will then be only re-sampled.
The snippets are oriented at the original images used for classification. Hence their size will be scale * 64x64, where the scale is set by the user. In this way, several scales can be searched efficiently on the same.
For every scale, a sliding window search is applied, where the snippet slides by `cells_per_step`

| Parameter			| Value |
|:-----------------:|:-----:|
|scales 			| 1.5, 1.75 (2)|
|cells per step 	| 2|

Here is the detection in a single image using three scales (three colors)

![alt text][image2]

### Classification
Every selected image snippet was then classified using the above trained SVC. In order to avoid false positive (detected cars in an empty snippet), a heatmap was created.
All pixel within a positive (true and false) window were added with 1 point to a heatmap.
That was done for all windows in the current image and the previous four images. Then a threshold was applied to remove false positives, which were usually not consistent over many windows and images.
Using `scipy.ndimage.measurements.label()` new bounding boxes (one for each car) were drawn on the image.

| Parameter				| Value |
|:---------------------:|:-----:|
| previous detections	| 4		|
| threshold				| 8 (12) 	|

Here is the same frame and its corresponding heatmap:

![alt text][image3]

### Result
The positive detection are then drawn on the images. Here is a [link to my video result](./Videos/video1_detected.mp4)

---

## Discussion

#### Problems
- Separation of cars: When two vehicles are close to each other, they will be detected as a single vehicle
- False positive: The pipeline does not reject all outliers
- The HOG calculation for the whole image runs incredible slow (10-15s per image). Reducing to only two scales and a narrow image stripe (y between 400 and 600) fastened the detection to ca. 5s per image but runs on the risk of missing vehicles now.
In facts the project videos misses the car now in several occations, re-running the video with other setting (lower threshold, and larger image stripe) would improve the detection, but the processing time is too long to repeat it.
On slack and [stackoverflow](https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python), it is suggested to run the OpenCV implementation rather than the scikit-image version as it would run 10x faster.

#### Improvements
- Image classification: The image classification could made more robust by performing a grid search tuning through a variety of hyper-parameters and also different classifier (e.g decision tree)
- Image classification: The data set could be augmented to improve prediction accuracy even further
- Sliding Window: In the current implementation the image is only separated into a part which is searched and a ignored part. In the whole searched part, the same scales are applied for the windows. An algorithm which decreased the scaled
towards the image borders would be beneficial in the sense of efficiency and robustness (e.g. a large search window in the upper left corner does not make sense).
- Combination with lane detection: In theory, the vehicle detection pipeline could be easily combined with the lane detection pipeline to give a more complete result. Also the information about the lane could be beneficial in order to check
the detection of vehicles: Are they driving on a lane? In which direction are they driving?
