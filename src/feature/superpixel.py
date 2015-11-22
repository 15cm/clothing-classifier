# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 4:14 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import os
import cv2
from mynp import np
from skimage.segmentation import slic
from skimage.util import img_as_float

CURPATH = os.path.split(os.path.realpath(__file__))[0]
DATAPATH = os.path.join(os.path.dirname(CURPATH),'dataset')

class SuperPixel:


    def __init__(self,image):
        self.img = cv2.imread(os.path.join(DATAPATH,image))

    def segment(self):
        self.segments = slic(img_as_float(self.img), enforce_connectivity=True)
        self.mask = np.zeros(self.img.shape[:2],dtype='int' )
        self.mask = self.mask - 1
        for (i, segVal) in enumerate(np.unique(self.segments)):
            self.mask[segVal == self.segments] = i




