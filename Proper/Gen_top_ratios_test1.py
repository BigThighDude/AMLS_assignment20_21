import numpy as np
import pickle
import Utilities
from os import path
from sklearn import svm
import os
import time
from sklearn.feature_selection import RFE
from sklearn import tree

def main(smile_gen, dir, perc):    # this function can be used for both smile and gender detection
    # declaring directory for the images and their labels
    startx = time.time()
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

    ratios = Utilities.lm_to_points(face_list, points, picked_lm, ratios)    # using the picked points calculate each of the ratios for each image

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

    print('Std Accuracy: ', accuracy)

    no_sel = 1
    no_wtd = no_ratios
    rfe = RFE(estimator=tree.DecisionTreeClassifier(), n_features_to_select=no_sel)
    rfe.fit(x_train, y_train)
    sel_ft = rfe.support_
    ranks = rfe.ranking_
    ranks = ranks.tolist()

    picked_lm2 = []
    for i in range(0, no_wtd):
        idx = ranks.index(i + 1) * 2
        picked_lm2.append(picked_lm[idx])
        picked_lm2.append(picked_lm[idx + 1])

    print(ranks)
    print(len(ranks))
    print(picked_lm)
    print(len(picked_lm2))


    min_lim = 0
    max_lim = no_ratios
    midp = int(max_lim/2)
    x = 0
    starty = time.time()
    acc_prev = accuracy
    max_acc_p = no_ratios
    while True:
        mov = -1
        x = x+1
        lm_temp = picked_lm2[:midp]
        lm_no = len(lm_temp)

        ratst = time.time()
        ratios_t = np.zeros([use_pics, lm_no], dtype=float)  # empty array to store the calculated ratios for each image
        ratios_t = Utilities.lm_to_points(face_list, points, lm_temp, ratios_t)
        ratse = time.time()
        print('ratio time: ', ratse-ratst)

        x_train2, y_train2, x_test2, y_exp2 = Utilities.data_split(perc, use_pics, face_list, ratios_t, y_use)

        clf = svm.SVC(gamma=0.01, C=1)
        clf. fit(x_train2, y_train2)
        accuracy2 = clf.score(x_test2, y_exp2)
        # y_test = clf.predict(x_test)
        print(end3-start3)
        print('RFE Accuracy: ', accuracy2)
        print(len(lm_temp))
        print(midp)
        if accuracy2>acc_prev:
            acc_prev = accuracy2
            max_acc_p = midp
            mov = mov
        elif accuracy2<acc_prev:
            mov = -1*mov
        if mov==1:
            min_lim = midp
        elif mov==-1:
            max_lim = midp
        midp = int((max_lim + min_lim) / 2)

        if x==7:
            break
    endy = time.time()
    print(endy-starty)
    print('max accuracy at: ',max_acc_p,' with accuracy of: ',acc_prev)

    # if accuracy>crit:
    #     svm_src = pkl_dir+str(picked_lm[0][0])+str(picked_lm[0][1])+str(picked_lm[1][0])+str(picked_lm[1][1])+'.pickle'
    #     with open(svm_src, 'wb') as f1:
    #         pickle.dump(clf, f1)  # save svm
    #     svm_src2 = pkl_dir+str(picked_lm[0][0])+str(picked_lm[0][1])+str(picked_lm[1][0])+str(picked_lm[1][1])+'points.pickle'
    #     with open(svm_src2, 'wb') as f2:
    #         pickle.dump(picked_lm, f2)  # save svm
    endx = time.time()
    print(endx-startx)



directory = './celeba'
smile_gen = 0
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

for i in range(0,1):
    main(smile_gen, directory, 0.75)
















