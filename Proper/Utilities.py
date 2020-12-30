import numpy as np
import cv2
import dlib
import pickle
import random
from csv import reader

def csv_in(src_csv):
    y_name = []  # names extracted from csv file appended to this list
    y_gen = []  # gender data extracted from csv file appended to this list
    y_smile = []  # smiling data extracted from csv file appended to this list

    with open(src_csv) as file:
        dat_read = reader(file, delimiter='\t')  # data is tab spaces
        temp = list(dat_read)[1:]  # remove first element of list (headers)
        for n in range(0, len(temp)):  # go through each element
            y_name.append(temp[n][1])  # second element is the name
            y_gen.append(temp[n][2])  # third element is gender
            y_smile.append(temp[n][3])  # fourth element is smiling or not
    return y_name, y_gen, y_smile

def process(no_pics, y_name, dir):
    src_img = dir + '/img/'  # directory for the images - file names will come later
    face_list = []
    points = np.array([[[None] * 2] * 68] * no_pics)  # 68 landmark coordinates stored here
    detector = dlib.get_frontal_face_detector()  # using dlib's face detector (doesn't work on angled faces)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # using dlib's 68 landmarks model

    for i in range(0, no_pics):
        # Open image and find face
        src = src_img + y_name[i]  # generating path for each image
        gray = cv2.imread(src, 0)  # image is read (using path) in grayscale - easier on cpu
        faces = detector(gray)  # run the frontal face detector on the grayscale image

        if list(faces):  # if a face is detected i.e. faces contains data
            face_list.append(i)  # keep track of iteration number in a list of detected faces
            face = list(faces)[0]  # takes data from first element of faces (as the detector can detect multiple faces)
            landmarks = predictor(gray, face)  # run the dlib 68 landmarks predictor on gray image in the face region
            for n in range(0, 68):  # for each of the landmarks
                points[i][n][0] = landmarks.part(n).x  # save the x coordinate of the nth landmark
                points[i][n][1] = landmarks.part(n).y  # save the y coordinate of the nth landmark

    points = list(points)
    face_list = list(face_list)
    points.append(face_list)
    # pickling data points for each image landmarks due to the time it takes to run each time
    with open(dir+'/datapoints.pickle', 'wb') as f1:
        pickle.dump(points, f1)  # dump points data to pickle file

def unpickle(dir):
    with open(dir+'/datapoints.pickle', 'rb') as f1:  # open the pickle file containing the landmark data
        temp1 = pickle.load(f1)  # load the file into a temporary variable
        points = temp1  # turn the temporary file into a list
        face_list = points[-1]  # extract the list of images where a face is detected
        points = points[0:-1]  # extract landmark data into one list
    return face_list, points

def pick_points(x):
    picked = []
    temp = list(range(0, 68))
    for i in range(0,int(x*2)):
        random.shuffle(temp)
        two_ps = temp[:2]
        picked.append(two_ps)
    return picked

def distance(two_ps):
    x = two_ps[0][0] - two_ps[1][0]
    y = two_ps[0][1] - two_ps[1][1]
    dist = np.sqrt(np.square(x)+np.square(y))
    return dist

def ratio(four_ps):
    d1 = distance(four_ps[:2])
    d2 = distance(four_ps[2:])
    if d2==0:
        rat = 0
    else:
        rat = d1/d2
    return rat

def lm_to_points(face_list, points, picked_lm, ratios):
    no_ratios = int(len(picked_lm)/2)
    use_pics = len(face_list)
    for idx in range(0, use_pics):
        m = face_list[idx]
        points_lm = []
        for i in range(0, no_ratios * 2):
            for n in range(0, 2):
                j = picked_lm[i][n]
                points_lm.append(list(points[m][j]))

        for i in range(0, no_ratios):
            x = i * 4
            ratios[idx][i] = ratio(points_lm[x:x + 4])
    return ratios

def data_split(perc, use_pics, face_list, ratios, y):
    y_use = []
    for i in range(0, use_pics):
        idx = face_list[i]
        y_use.append(y[idx])

    split = int(np.floor(perc * use_pics))

    x_train = ratios[:split]
    y_train = y_use[:split]
    x_test = ratios[split:]
    y_exp = y_use[split:]
    return x_train, y_train, x_test, y_exp






























