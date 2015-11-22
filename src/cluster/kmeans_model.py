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
        self.model_path = os.path.join(CURPATH,'model')

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
        return self.kmeans.predict(X)

    def save(self,model_name):
        joblib.dump(self.kmeans,os.path.join(self.model_path,model_name + '.pkl'))

    def load(self,model_name):
        model_file = os.path.join(self.model_path,model_name + '.pkl')
        if os.path.exists(model_file):
            self.kmeans = joblib.load(model_file)
            self.n_clusters = self.kmeans.n_clusters
            return True
        else:
            return False
