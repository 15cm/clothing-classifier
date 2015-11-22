# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 8:37 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

from feature.sift import Sift
from cluster.kmeans_model import KmeansModel
from classifier.clf_model import RandomForest
from cluster.bag_of_words import Bow
from mynp import np
import os

CURPATH = os.path.split(os.path.realpath(__file__))[0]
DATAPATH = os.path.join(os.path.dirname(CURPATH),'dataset')

def train_bow(bow_name,id_upper):
    if not id_upper:
        id_upper = 3900
    sift = Sift()
    kmeans = KmeansModel()
    image_list = sorted([x for x in os.listdir(DATAPATH) if os.path.splitext(x)[1] == '.jpg' and int(os.path.splitext(x)[0]) < id_upper],key=lambda x: int(os.path.splitext(x)[0]))
    descriptors_list = sift.compute(image_list)
    kmeans.fit(descriptors_list)
    kmeans.save(bow_name)


    bow = Bow(kmeans)
    # bag of words of samples
    # label indicator1 indicator2 ...
    # ...   ...        ...        ...
    bow.train(descriptors_list,bow_name)
    # labels of samples

def train_clf(bow_name):
    data = np.loadtxt(bow_name)
    X = data[:,1:]
    y = data[:,0]
    clf = RandomForest()
    clf.fit(X,y)
    clf.save()


