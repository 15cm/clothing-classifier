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
from feature.sift import Sift

CURPATH = os.path.split(os.path.realpath(__file__))[0]
DATAPATH = os.path.join(os.path.dirname(CURPATH),'dataset')

class SuperPixel:

    class Pixel:

        def __init__(self,id):
            self.id = id
            self.descriptor_list = []

        def add_descriptor(self,descriptor):
            self.descriptor_list.append(descriptor)

        def num(self):
            return len(self.descriptor_list)

    def __init__(self,image):
        self.img = cv2.imread(os.path.join(DATAPATH,image))
        self.pixel_list = []

    def segment(self):
        self.segments = slic(img_as_float(self.img), enforce_connectivity=True)
        self.mask = np.zeros(self.img.shape[:2],dtype='int' )
        self.mask = self.mask - 1
        for (i, segVal) in enumerate(np.unique(self.segments)):
            self.mask[segVal == self.segments] = i
            self.pixel_list.append(self.Pixel(i))

    def count_descriptors(self):
        sift = Sift()
        kd_des = sift.detect_and_compute(self.img)
        for i in range(len(kd_des[0])):
            kd = kd_des[0][i]
            id = self.mask[int(kd[0][0]),kd[0][1]]
            if id != -1:
                self.pixel_list[id].add_descriptor(kd_des[1][i])
        pixel_num = len(self.pixel_list)
        self.pixel_list = sorted(self.pixel_list,key=lambda x: x.num(),reverse=True)[:int(pixel_num/2)]




