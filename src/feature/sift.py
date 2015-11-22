# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 4:13 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import cv2


class Sift:

    def __init__(self,edgeThreshold=0.2,sigma=3):
        self.sift = cv2.SIFT(edgeThreshold=edgeThreshold,sigma=sigma)

    def compute(self,image):
        gray_im = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2GRAY)
        return self.sift.Compute(gray_im,None)
