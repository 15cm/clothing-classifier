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

    def count_descriptors(self,sift):
        kp_list = sift.keypoints_list
        des_list = sift.descriptors_list
        for i in range(len(kp_list)):
            id = self.mask[int(kp_list[i]),int(kp_list)]
            if id != -1:
                self.pixel_list[id].add_descriptor(des_list[i])
        pixel_num = len(self.pixel_list)
        self.pixel_list = sorted(self.pixel_list,key=lambda x: x.num(),reverse=True)[:int(pixel_num/2)]
        self.pixel_list = list(filter(lambda x: x.num() > 0,self.pixel_list))
        print 'ok'




