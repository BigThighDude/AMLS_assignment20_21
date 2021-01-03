import pickle
import os
from csv import reader
from sklearn import svm
import dlib
import cv2
import numpy as np

def main(sel, model):
    if sel==0:
        print("Task A1: Gender Detection")
        model = create_model()

        return model
    elif sel==1:
        directory = 'celeba/'
        y_name, y_gen, y_smile, face_list, points = import_data(directory)
        print("No. of faces detected in this set: ", len(face_list))
        x_use, y_use = process(points, face_list, y_gen)
        perc = 0.8
        x_train, y_train, x_vald, y_vald = split_data(perc, x_use, y_use)
        print("Number of training samples: ", len(x_train))
        model = train_model(model, x_train, y_train)
        print("Number of validation samples: ", len(x_vald))
        accuracy = test_model(model, x_vald, y_vald)
        print("Accuracy of validation set: ", str(accuracy), "\n")

        return accuracy
    elif sel==2:
        directory = 'celeba_test/'
        y_name, y_gen, y_smile, face_list, points = import_data(directory)
        print("No. of faces detected in this set: ", len(face_list))
        x_use, y_use = process(points, face_list, y_gen)
        print("Number of test samples: ", len(x_use))
        accuracy = test_model(model, x_use, y_use)
        print("Accuracy of unseen test set: ", str(accuracy), "\n")

        return accuracy

def create_model():
    print("Creating model...")
    clf = svm.SVC(kernel='poly', degree=3)

    return clf

def split_data(perc, x_use, y_use):
    split = int(perc*len(x_use))
    x_train = x_use[:split]
    y_train = y_use[:split]
    x_vald = x_use[split:]
    y_vald = y_use[split:]

    return x_train, y_train, x_vald, y_vald

def train_model(model, x_train, y_train):
    print("Training model...")
    model = model.fit(x_train, y_train)
    print("Model training finished")

    return model


def test_model(model, x_t, y_t):
    print("Testing model...")
    accuracy = model.score(x_t, y_t)
    print("Model testing finished")

    return accuracy

################### Flatten Raw Coordinates ###################
def process(data, face_list, y_use):
    use_pic = len(face_list)
    dat_pr = []
    y_proc = []
    for i in range(0, use_pic):
        idx = face_list[i]
        temp = []
        for m in range(0, 68):
            for n in range(0, 2):
                temp.append(data[idx][m][n])
        dat_pr.append(temp)
        y_proc.append(y_use[idx])

    return dat_pr, y_proc

def import_data(directory):
    print("Acquiring labels and landmark data...")
    full_dir = str(os.path.dirname(__file__)[:-2])+'/Datasets/'+directory
    csv_src = os.path.join(full_dir, "labels.csv")
    img_src = os.path.join(full_dir, "img/")

    y_name = []
    y_gen = []
    y_smile = []
    face_list = []
    with open(csv_src) as file:
        dat_read = reader(file, delimiter='\t')  # data is tab spaces
        temp = list(dat_read)[1:]  # remove first element of list (headers)
        for n in range(0, len(temp)):  # go through each element
            y_name.append(temp[n][1])  # second element is the name
            y_gen.append(temp[n][2])  # third element is gender
            y_smile.append(temp[n][3])  # fourth element is smiling or not
    no_pics = len(y_name)

    if not os.path.isfile(full_dir+'facial_landmarks.pickle'):  # if data file containing facial landmarks doesnt exist
        print("Landmark data not found - generating...")
        detector = dlib.get_frontal_face_detector()  # using dlib's face detector (doesn't work on angled faces)
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # using dlib's 68 landmarks model
        points = np.array([[[None] * 2] * 68] * no_pics)  # 68 landmark coordinates stored here

        for i in range(0, no_pics): # Open image and find face
            src = img_src + y_name[i]  # generating path for each image
            gray = cv2.imread(src, 0)  # image is read (using path) in grayscale - easier to process
            faces = detector(gray)  # run the frontal face detector on the grayscale image

            if list(faces):  # if a face is detected i.e. faces contains data
                face_list.append(i)  # keep track of iteration number in a list of detected faces
                face = list(faces)[0]  # takes data from first element of faces (the detector can detect multiple faces)
                landmarks = predictor(gray, face)  # run the dlib 68 landmarks predictor on gray image in face region
                for n in range(0, 68):  # for each of the landmarks
                    points[i][n][0] = landmarks.part(n).x  # save the x coordinate of the nth landmark
                    points[i][n][1] = landmarks.part(n).y  # save the y coordinate of the nth landmark

        points = list(points)
        face_list = list(face_list)
        points.append(face_list)
        # pickling data points for each image landmarks due to the time it takes to run each time
        with open(full_dir+'facial_landmarks.pickle', 'wb') as f1:
            pickle.dump(points, f1)  # dump points data to pickle file

    with open(full_dir+'facial_landmarks.pickle', 'rb') as f2:  # open the pickle file containing the landmark data
        temp1 = pickle.load(f2)  # load the file into a temporary variable
        points = temp1  # turn the temporary file into a list
        face_list = points[-1]  # extract the list of images where a face is detected
        points = points[0:-1]  # extract landmark data into one list

    return y_name, y_gen, y_smile, face_list, points



