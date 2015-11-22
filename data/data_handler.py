# -*- coding: utf-8 -*-
'''
Created by 15 cm on 11/22/15 3:20 PM
Copyright © 2015 15cm. All rights reserved.
'''
__author__ = '15cm'

import json
import urllib2
import multiprocessing
import numpy as np
from PIL import Image
import io
import os

def download_stuff(stuff):
    image_bytes = urllib2.urlopen(stuff.link).read()
    data_stream = io.BytesIO(image_bytes)
    pil_image = Image.open(data_stream)
    try:
        pil_image.load()
    except IOError:
        pass
    w,h = pil_image.size
    pil_image.thumbnail((w/3,h/3))
    pil_image.save(os.path.join('data',str(stuff.id) + '.jpg'),'jpeg')



class DataHandler:

    class ImageData:
        def __init__(self,id,link,label):
            self.id = id
            self.link = link
            self.label = label

    def __init__(self):
        self.data = [] # [(link,label),...]
        self.label_dict = {}
        self.label_list = []
        self.data_file = os.path.join('data','data.txt')

    def label_filter(self,s):
        # valid_word_list = ['衣','裙','裤','长','大','短','单','套','衫','毛']
        valid_word_list = ['裙','衣','裤']
        valid_word_set = set((map(lambda x: x.decode('utf-8'),valid_word_list)))
        res_str = ''
        if not isinstance(s,unicode):
            s = s.decode('utf-8')
        for word in s:
            if word in valid_word_set:
                res_str += word
                break
        if not res_str:
            res_str = '其他'.decode('utf-8')
        return res_str.encode('utf-8')

    def parse_data(self,json_file):
        with open(json_file) as f:
            json_content = json.load(f)
            for item in json_content:
                id=int(item['id'])
                label = self.label_filter(item['sub_category'])
                link = item['picture']
                if not self.label_dict.has_key(label):
                    self.label_list.append(label)
                    self.label_dict[label] = len(self.label_list) - 1
                self.data.append(self.ImageData(id, link, self.label_dict[label]))

    def download(self,num = -1,id_geq = 0):
        if num > 0:
            data = [x for x in self.data if x.id < num and x.id > id_geq]
        else:
            data = [x for x in self.data if x.id > id_geq]
        pool = multiprocessing.Pool(processes=5)
        pool.map(download_stuff,data)

    def save(self):
        # data_matrix:
        # id    label
        # ...   ...
        data_matrix = np.empty((len(self.data),2))
        for i in range(len(self.data)):
            data_matrix[i][0] = self.data[i].id
            data_matrix[i][1] = self.data[i].label
        np.savetxt(self.data_file,data_matrix)
        with open('label_list.json','w') as f:
            json.dump(self.label_list,f)



    def load(self):
        self.data_matrix = np.loadtxt(self.data_file)
        with open('label_list.json') as f:
            self.label_list = json.load(f)


    def get_lables(self,id = -1):
        if id >= 0:
            return self.data_matrix[id][1]
        else:
            return self.data_matrix[:,1]

    def tell_label(self,label):
        return self.label_list[label]








