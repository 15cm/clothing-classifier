# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 4:13 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import cv2
import os

CURPATH = os.path.split(os.path.realpath(__file__))[0]
DATAPATH = os.path.join(os.path.dirname(CURPATH),'dataset')


class Sift:

    def __init__(self,edgeThreshold=0.2,sigma=3):
        self.sift = cv2.SIFT(edgeThreshold=edgeThreshold,sigma=sigma)

    def compute(self,file):
        if hasattr(file,'__iter__'):
            descriptors_list = []
            for f in file:
                f = os.path.join(DATAPATH,f)
                gray_im = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY)
                descriptors_list.append(self.sift.detectAndCompute(gray_im,None)[1])
            return descriptors_list
        else:
            f = os.path.join(DATAPATH,file)
            gray_im = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY)
            return self.sift.detectAndCompute(gray_im,None)[1]


