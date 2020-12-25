import numpy as np
import cv2
import dlib
import time
import pickle
import utilities
from csv import reader
from sklearn import svm

no_pics = 5000
y_smile = [None]*no_pics

with open('datapoints.pickle', 'rb') as f:
    points = pickle.load(f)
points = list(points)
with open('./celeba/labels.csv') as file:
    datread = reader(file, delimiter='\t')
    temp = list(datread)
    temp = temp[1:]
    # print(temp)
    for i in range(0, no_pics):
        y_smile[i] = temp[i][3]

feat_nos = []

