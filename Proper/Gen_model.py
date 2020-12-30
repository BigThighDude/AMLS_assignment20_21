import numpy as np
import pickle
import Utilities
from os import path
from sklearn import svm
import os
import time

def main(smile_gen, dir, perc):    # this function can be used for both smile and gender detection
    # declaring directory for the images and their labels
    start1 = time.time()
    src_csv = dir+'/labels.csv'     # directory and file name for the image labels

    y_name, y_gen, y_smile = Utilities.csv_in(src_csv)  # use our function to extract data from the csv file
    no_pics = len(y_name)    # tells us how many images there are in total

    if not path.isfile(dir+'/datapoints.pickle'):  # if the data file containing facial landmark info doesnt exist
        Utilities.process(no_pics, y_name, dir)     # run function to get facial landmarks and save to pickle file

    face_list, points = Utilities.unpickle(dir)    # get the list of usable faces and landmark data from the pickle file

    use_pics = len(face_list)   # the number of usable faces
    no_ratios = 100      # the number of distance ratios we want to use, we can change this as we like
    ratios = np.zeros([use_pics, no_ratios], dtype=float)   # empty array to store the calculated ratios for each image
    picked_lm = Utilities.pick_points(no_ratios)  # pick 4 random points per ratio to calculate for each image
    # picked_lm = [[24, 57], [55, 58], [5, 31], [48, 66], [48, 13], [64, 15], [63, 66], [20, 56], [64, 47], [39, 20], [67, 31], [60, 65], [37, 2], [30, 48], [6, 17], [48, 29], [51, 53], [36, 11], [38, 48], [6, 51], [35, 55], [12, 56], [65, 18], [18, 37], [21, 56], [40, 0], [37, 41], [49, 25], [58, 43], [39, 46], [51, 54], [59, 23], [4, 13], [27, 63], [63, 58], [23, 24], [47, 18], [18, 19], [20, 34], [13, 8], [25, 26], [26, 64], [52, 34], [25, 38], [60, 25], [29, 8], [37, 13], [34, 32]]
    end1 = time.time()
    print(end1-start1)
    start2 = time.time()
    ratios = Utilities.lm_to_points(face_list, points, picked_lm, ratios)    # using the picked points calculate each of the ratios for each image
    end2 = time.time()
    print(end2-start2)

    start3 = time.time()
    if smile_gen==0:
        y_use = y_smile
        pkl_dir = './trained_smile/'
        crit = 0.8
    elif smile_gen==1:
        y_use = y_gen
        pkl_dir = './trained_gen/'
        crit = 0.8

    # get the x and y data for training and testing from the calculated rations and the data in the csv file
    x_train, y_train, x_test, y_exp = Utilities.data_split(perc, use_pics, face_list, ratios, y_use)

    clf = svm.SVC(gamma=0.01, C=1)
    clf. fit(x_train, y_train)
    accuracy = clf.score(x_test, y_exp)
    # y_test = clf.predict(x_test)
    end3 = time.time()
    print(end3-start3)
    print(accuracy)
    print(use_pics)
    # if accuracy>crit:
    #     svm_src = pkl_dir+str(picked_lm[0][0])+str(picked_lm[0][1])+str(picked_lm[1][0])+str(picked_lm[1][1])+'.pickle'
    #     with open(svm_src, 'wb') as f1:
    #         pickle.dump(clf, f1)  # save svm
    #     svm_src2 = pkl_dir+str(picked_lm[0][0])+str(picked_lm[0][1])+str(picked_lm[1][0])+str(picked_lm[1][1])+'points.pickle'
    #     with open(svm_src2, 'wb') as f2:
    #         pickle.dump(picked_lm, f2)  # save svm


directory = './celeba'
smile_gen = 1
# if smile_gen==0:
#     mod_dir = './trained_smile'
# elif smile_gen==1:
#     mod_dir = './trained_gen'
#
# no_wtd = 13
# no_models = int(len(os.listdir(mod_dir))/2)
#
# while no_models<no_wtd:
#     main(smile_gen, directory, 0.75)
#     no_models = int(len(os.listdir(mod_dir)) / 2)

main(smile_gen, directory, 0.75)















