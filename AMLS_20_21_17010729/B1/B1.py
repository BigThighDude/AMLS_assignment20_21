import os
from csv import reader
import pickle
import numpy as np
import cv2
from sklearn import svm

def main(sel, model):
    if sel==0:
        print("Task B1: Face Shape Recognition")
        model = create_model()

        return model
    elif sel==1:
        directory = 'cartoon_set/'
        y_name, y_eye, y_shp, points = import_data(directory)
        x_use = process(points)
        perc = 0.8
        x_train, y_train, x_vald, y_vald = split_data(perc, x_use, y_shp)
        print("Number of training samples: ", len(x_train))
        model = train_model(model, x_train, y_train)
        print("Number of validation samples: ", len(x_vald))
        accuracy = test_model(model, x_vald, y_vald)
        print("Accuracy of validation set: ", str(accuracy), "\n")

        return accuracy
    elif sel==2:
        directory = 'cartoon_set_test/'
        y_name, y_eye, y_shp, points = import_data(directory)
        x_use = process(points)
        print("Number of test samples: ", len(y_shp))
        accuracy = test_model(model, x_use, y_shp)
        print("Accuracy of unseen test set: ", str(accuracy), "\n\n\n")

        return accuracy

def create_model():
    print("Creating model...")
    clf = svm.SVC(kernel='poly')

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

def process(points):
    no_pics = len(points[0])
    points_sort = []
    for i in range(0, no_pics):
        temp = []
        temp.append(points[0][i][0])
        temp.append(points[1][i][0])
        temp.append(points[2][i][0])
        points_sort.append(temp)

    return points_sort

def import_data(directory):
    print("Acquiring labels and slice data...")
    full_dir = str(os.path.dirname(__file__)[:-2])+'Datasets/'+directory
    csv_src = os.path.join(full_dir, "labels.csv")
    img_src = os.path.join(full_dir, "img")

    with open(csv_src) as file:
        dat_read = list(reader(file, delimiter='\t'))
        labels = list(dat_read)[1:]

    no_pics = len(labels)
    y_name = []
    y_eye = []
    y_shp = []
    for i in range(0, no_pics):
        y_name.append(labels[i][3])
        y_eye.append(labels[i][1])
        y_shp.append(int(labels[i][2]))

    if not os.path.isfile(full_dir+'fshp_slice.pickle'):
        print("Sample data not found - generating...")
        slice_points = [[[160, 365], [250, 365]], [[130, 335], [215, 335]], [[157, 290], [173, 290]]]
        no_slice = len(slice_points)
        points = np.zeros(shape=(no_slice, no_pics, 1))
        for m in range(0, no_pics):
            for n in range(0, no_slice):
                img = cv2.imread(os.path.join(img_src,y_name[m]))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                slice_d = slice_points[n][0][1]
                slice_s = slice_points[n][0][0]
                slice_e = slice_points[n][1][0]
                sample_n = img[:][slice_d]
                sample_n = sample_n[slice_s:slice_e]
                avg = np.zeros(shape=(len(sample_n), 1), dtype=float)
                for i in range(0, len(sample_n)):
                    avg[i] = np.mean(sample_n[i])
                min_val = avg.min()
                pos_list = np.where(avg == min_val)
                pos_mid = pos_list[0][int(len(pos_list[0]) / 2)]
                avg_at_pos = avg[pos_mid]
                if avg_at_pos < 35:
                    points[n][m] = int(pos_mid)
                else:
                    if n == 0:
                        points[n][m] = 53.4
                    elif n == 1:
                        points[n][m] = 52.0
                    elif n == 2:
                        points[n][m] = 9.2
                    # points[n][m] = int(len(sample_n)*0.5)
        with open(full_dir+'fshp_slice.pickle', 'wb') as f1:
            pickle.dump(points, f1)

    with open(full_dir+'fshp_slice.pickle', 'rb') as f2:
        points = pickle.load(f2)

    return y_name, y_eye, y_shp, points
