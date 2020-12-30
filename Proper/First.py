import numpy as np
import pickle
import Utilities
from os import path
from sklearn import svm

def main(smile_gen, dir):    # this function can be used for both smile and gender detection
    # declaring directory for the images and their labels
    src_img = dir+'/img/'    # directory for the images - file names will come later
    src_csv = dir+'/labels.csv'     # directory and file name for the image labels

    y_name, y_gen, y_smile = Utilities.csv_in(src_csv)  # use our function to extract data from the csv file
    no_pics = len(y_name)    # tells us how many images there are in total

    if not path.isfile(dir+'/datapoints.pickle'):  # if the data file containing facial landmark info doesnt exist
        Utilities.process(no_pics, src_img, y_name, dir)     # run function to get facial landmarks and save to pickle file

    face_list, points = Utilities.unpickle(dir)    # get the list of usable faces and landmark data from the pickle file

    use_pics = len(face_list)   # the number of usable faces
    no_ratios = 12      # the number of distance ratios we want to use, we can change this as we like
    ratios = np.zeros([use_pics, no_ratios], dtype=float)   # empty array to store the calculated ratios for each image

    picked_lm = Utilities.pick_points(no_ratios*4)  # pick 4 random points per ratio to calculate for each image
    # picked_lm = [[36, 39], [0, 16], [42, 45], [0, 16], [51, 57], [48, 54], [36, 39], [48, 54], [42, 45], [48, 54], [0, 16], [8, 27]]
    ratios = Utilities.lm_to_points(face_list, no_ratios, points, picked_lm, ratios)    # using the picked points calculate each of the ratios for each image
    perc = 0.75     # the percentage split for training and testing images

    if smile_gen==0:
        y_use = y_smile
    elif smile_gen==1:
        y_use = y_gen

    # get the x and y data for training and testing from the calculated rations and the data in the csv file
    x_train, y_train, x_test, y_exp = Utilities.data_split(perc, use_pics, face_list, ratios, y_use)

    clf = svm.SVC(gamma=0.01, C=1)
    clf. fit(x_train, y_train)
    accuracy = clf.score(x_test, y_exp)
    # y_test = clf.predict(x_test)

    print(accuracy)
    print(picked_lm)


    if accuracy>0.8:
        svm_src = './trained_smile/'+str(picked_lm[0][0])+str(picked_lm[0][1])+str(picked_lm[1][0])+str(picked_lm[1][1])+'.pickle'
        with open(svm_src, 'wb') as f1:
            pickle.dump(clf, f1)  # save svm
        svm_src2 = './trained_smile/'+str(picked_lm[0][0])+str(picked_lm[0][1])+str(picked_lm[1][0])+str(picked_lm[1][1])+'points.pickle'
        with open(svm_src2, 'wb') as f2:
            pickle.dump(picked_lm, f2)  # save svm



    # print(points_lm[0:2])
    # print(ratios)
    # print(ratios)
    # print(picked_lm)
    # print(points[0])
    # print(len(points_lm))
    # print(ratios[0])
    # print(points_lm)

directory = './celeba'
main(0, directory)
















