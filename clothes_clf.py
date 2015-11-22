# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 2:55 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import argparse
from data.train import train_bow_sift,train_bow_pixel,train_clf
from data.data_handler import DataHandler
from cluster.kmeans_model import KmeansModel
from classifier.clf_model import RandomForest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='clothes_parser.py',description='A Clothes Classifier')
    parser.add_argument('cmd',action='store',type=str,help='train,data,test')
    parser.add_argument('-b',action='store_true',help='train bag of words')
    parser.add_argument('-s',action='store_true',help='train bag of words by sift')
    parser.add_argument('-p',action='store_true',help='train bag of words by superpixel')
    parser.add_argument('-i',action='store',dest='id_upper',type = int)
    parser.add_argument('-c',action='store_true',help='train classifier')
    parser.add_argument('-f',action='store',dest='file',type=str,help='parser a clothes image')
    args = parser.parse_args()


    if args.cmd == 'train':
        if args.b:
            if args.s:
                train_bow_sift(args.id_upper)
            elif args.p:
                train_bow_pixel()
        if args.c:
            train_clf('pixel')

    if args.cmd == 'test':
        kmeans = KmeansModel()
        kmeans.load('kmeans_pixel')
        clf = RandomForest()
        clf.load()
        data = DataHandler()
        data.load()
        if args.file:
            for res in clf.predict(kmeans,file):
                print int(res),data.tell_label(int(res))

    if args.cmd == 'data':
        data = DataHandler()
        data.parse_data('design.json')
        data.save()
