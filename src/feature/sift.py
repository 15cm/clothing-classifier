# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 4:13 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import cv2
import os
import pickle

CURPATH = os.path.split(os.path.realpath(__file__))[0]
DATAPATH = os.path.join(os.path.dirname(CURPATH),'dataset')


class Sift:

    def __init__(self,edgeThreshold=0.2,sigma=3):
        self.sift = cv2.SIFT(edgeThreshold=edgeThreshold,sigma=sigma)
        self.data_file = os.path.join(CURPATH,'sift.txt')
        # self.kp_file = os.path.join(CURPATH,'kp.json')
        # self.des_file = os.path.join(CURPATH,'des.json')

    def compute(self,file):

        if hasattr(file,'__iter__'):
            self.descriptors_list = []
            self.keypoints_list = []
            for f in file:
                f = os.path.join(DATAPATH,f)
                gray_im = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY)
                self.keypoints_list.append(self.sift.detectAndCompute(gray_im,None)[0])
                self.descriptors_list.append(self.sift.detectAndCompute(gray_im,None)[1])
        else:
            f = os.path.join(DATAPATH,file)
            gray_im = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY)
            self.keypoints_list,self.descriptors_list = self.sift.detectAndCompute(gray_im,None)

    def detect_and_compute(self,img):
        gray_im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray_im,None)

    # def save(self):
    #     with open(self.data_file,'wb') as f:
    #         pickle.dump((self.keypoints_list,self.descriptors_list),f)
    #     # with open(self.kp_file,'w') as f:
    #     #     json.dump(self.keypoints_list,f)
    #     # with open(self.des_file,'w') as f:
    #     #     json.dump(self.descriptors_list,f)
    #
    # def load(self):
    #     if not os.path.exists(self.data_file):
    #         return False
    #     # if not os.path.exists(self.kp_file) or not os.path.exists(self.des_file):
    #     #     return False
    #     else:
    #         with open(self.data_file,'rb') as f:
    #             temp = pickle.load(f)
    #             self.keypoints_list = temp[0]
    #             self.descriptors_list = temp[1]
    #         # with open(self.kp_file) as f:
    #         #     self.keypoints_list = json.load(f)
    #         # with open(self.des_file) as f:
    #         #     self.descriptors_list = json.load(f)
    #         return True



