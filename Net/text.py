#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:36:18 2019

@author: bozhao
"""
import os
import numpy as np

filenames = os.listdir('../data/ResizedData/train_data')
filenames = [os.path.join('../data/ResizedData/train_data', f) for f in filenames if f.endswith('.png')]

labels = []
for filename in filenames:
    imagename = os.path.split(filename)[-1]
    ind = imagename.find('(')
    label = imagename[(ind+1):(ind+3)]
    finallabel = np.array(list(map(int, list(label))))
    labels.append(finallabel)

print(labels[123])