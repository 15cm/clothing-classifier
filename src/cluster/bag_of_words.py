# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 6:33 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

from data.data_handler import DataHandler
from mynp import np
import os

CURPATH = os.path.split(os.path.realpath(__file__))[0]

class Bow:

    def __init__(self,kmeans):
        self.kmeans = kmeans
        self.bow_path =os.path.join(os.path.join(CURPATH,'bow'),'bow.txt')

    def train(self,X_list):
        bow_list = []
        for X in X_list:
            bow_list.append(self.kmeans.predict(X))
        self.bow_matrix = reduce(np.vstack,bow_list)
        dh = DataHandler()
        dh.load()
        sample_y = np.empty((len(X_list),1))
        for i in range(len(sample_y)):
            sample_y[i][0] = dh.get_lables(id=i)
        sample_data = np.hstack(sample_y,self.bow_matrix)
        # save sample data
        np.savetxt(self.bow_path,sample_data)

    def compute(self,kmeans,X):
        bow = [0 for x in range(kmeans.n_cluster)]
        clusters = self.kmeans.predict(X)
        for i in clusters:
            bow[i] += 1
        return bow

    def load(self):
        self.bow_matrix = np.loadtxt(self.bow_path)




