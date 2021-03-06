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

def train_bow_sift(id_upper):
    if not id_upper:
        id_upper = 3900
    sift = Sift()
    kmeans = KmeansModel()
    image_list = sorted([x for x in os.listdir(DATAPATH) if os.path.splitext(x)[1] == '.jpg' and int(os.path.splitext(x)[0]) < id_upper],key=lambda x: int(os.path.splitext(x)[0]))
    sift.compute(image_list)
    if not kmeans.load('kmeans_sift'):
        kmeans.fit(sift.descriptors_list)
        kmeans.save('kmeans_sift')

    bow = Bow(kmeans)
    # bag of words of samples
    # label indicator1 indicator2 ...
    # ...   ...        ...        ...
    bow.train_sift(sift.descriptors_list)
    # labels of samples

def train_bow_pixel():
    kmeans_sift = KmeansModel()
    sift = Sift()
    if not kmeans_sift.load('kmeans_sift'):
        print 'kmeans_sift not found!'
    if not sift.load():
        print 'no sift data found!'
        exit(1)





def train_clf(bow_name):
    data = np.loadtxt(bow_name)
    X = data[:,1:]
    y = data[:,0]
    clf = RandomForest()
    clf.fit(X,y)
    clf.save()


