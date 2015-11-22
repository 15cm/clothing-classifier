# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 2:55 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='clothes_parser.py',description='A Clothes Classifier')
    parser.add_argument('cmd',action='store',type=str,help='train,data,test')
    parser.add_argument('-b',action='store_true',help='train bag of words')
    parser.add_argument('-i',action='store',dest='id_upper',type = int)
    parser.add_argument('-c',action='store_true',help='train classifier')
    parser.add_argument('-f',action='store',dest='file',type=str,help='parser a clothes image')
    args = parser.parse_args()
