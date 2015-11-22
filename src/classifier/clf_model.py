# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 3:28 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from feature.sift import Sift
import os

CURPATH = os.path.split(os.path.realpath(__file__))[0]

class RandomForest:
    def __init__(self):
        self.rf = RandomForestClassifier()
        self.model_path =os.path.join(os.path.join(CURPATH,'model'),'random_forest.pkl')

    def fit(self,X,y):
        self.rf.fit(X,y)

    def save(self):
        joblib.dump(self.rf,self.model_path)

    def load(self):
        self.rf = joblib.load(self.model_path)

    # def predict(self,kmeans,file):
    #     if image:
    #         X = bow.compute_bow(kmeans,image)
    #     elif image_list:
    #         descriptors_list = [sift.get_descriptors(os.path.join('test',x)) for x in image_list] # 128-dimension descriptors for every image
    #         X = bow.compute_bow_matrix(kmeans,descriptors_list)
    #     else:
    #         print 'image or image_list should be provided'
    #         exit(1)
    #     return self.rf.predict(X)
