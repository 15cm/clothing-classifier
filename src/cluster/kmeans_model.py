# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 3:27 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import os
from mynp import np
from feature.sift import Sift
from sklearn.cluster import KMeans
from sklearn.externals import joblib

CURPATH = os.path.split(os.path.realpath(__file__))[0]

class KmeansModel:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=15,n_init=15)
        self.model_path =os.path.join(os.path.join(CURPATH,'model'),'kmeans.pkl')

    def fit(self,X):
        if type(X) == list:
            self.kmeans.fit(reduce(np.vstack,X))
        else:
            self.kmeans.fit(X)
        self.n_clusters = self.kmeans.n_clusters


    @property
    def n_cluster(self):
        return self.n_clusters

    def predict(self,X):
        if type(X) == str:
            sift = Sift()
            return self.kmeans.predict(sift.compute(X))
        else:
            return self.kmeans.predict(X)

    def save(self):
        joblib.dump(self.kmeans,self.model_path)

    def load(self):
        self.kmeans = joblib.load(self.model_path)
        self.n_clusters = self.kmeans.n_clusters
