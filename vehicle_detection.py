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
from moviepy.editor import VideoFileClip

import lane as l

#importing own functions
try:
    DIST = dist.mtx
    MTX = dist.mtx
except:
    import undistort as dist
    DIST = dist.mtx
    MTX = dist.mtx