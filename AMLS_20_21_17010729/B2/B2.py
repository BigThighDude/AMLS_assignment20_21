import os
from csv import reader
import pickle
import numpy as np
import cv2
from sklearn import svm

def main(sel, model):
    if sel==0:
        print("Task B2: Eye Colour Recognition")
        model = create_model()

        return model
    elif sel==1:
        directory = 'cartoon_set/'
        y_name, y_eye, y_shp, all_index, list_col = import_data(directory)
        x_use, y_use = process(all_index, list_col, y_eye)
        perc = 0.8
        print("Number of non-obscured eyes: ", len(y_use))
        x_train, y_train, x_vald, y_vald = split_data(perc, x_use, y_use)
        print("Number of training samples: ", len(x_train))
        model = train_model(model, x_train, y_train)
        print("Number of validation samples: ", len(x_vald))
        accuracy = test_model(model, x_vald, y_vald)
        print("Accuracy of validation set: ", str(accuracy), "\n")

        return accuracy
    elif sel==2:
        directory = 'cartoon_set_test/'
        y_name, y_eye, y_shp, all_index, list_col = import_data(directory)
        x_use, y_use = process(all_index, list_col, y_eye)
        print("Number of non-obscured eyes: ", len(y_use))
        print("Number of test samples: ", len(y_use))
        accuracy = test_model(model, x_use, y_use)
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

def process(all_index, list_col, y_sel):
    use_list = []
    max_fn = len(list_col)
    for i in range(0, max_fn):
        if all_index[i]==1 or all_index[i]==0:
            use_list.append(i)

    use_no = len(use_list)
    x_use = []
    y_use = []
    for i in range(0, use_no):
        idx = use_list[i]
        x_use.append(list_col[idx])
        y_use.append(y_sel[idx])

    return x_use, y_use

def col_proc(sample):
    mix_col = sample[-1]
    tint_offset = sample[14]
    col = (mix_col-tint_offset)*2.9

    return col

def import_data(directory):
    print("Acquiring labels and slice data...")
    full_dir = str(os.path.dirname(__file__)[:-2])+'Datasets/'+directory
    csv_src = os.path.join(full_dir, "labels.csv")
    img_src = os.path.join(full_dir, "img/")

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

    if not os.path.isfile(full_dir+'eyec_data.pickle'):
        print("Sample data not found - generating...")
        white = np.array([255,255,255])
        ovr_l = []
        col_list = []
        max_fn = len(y_name)
        for i in range(0, max_fn):
            img = cv2.imread(img_src + y_name[i])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            sample_n = img[:][263]
            sample_n = sample_n[190:216]  # 190, 216 ## 204 is black i.e. 14
            mpi = sample_n[int(len(sample_n) / 2)]
            if np.allclose(mpi, sample_n):  # Obscured
                ovr_l.append(-1)
                col_list.append(None)
            else:
                counter = 0
                for n in range(0, len(sample_n)):
                    ans = (sample_n[n] == white).all()
                    if ans:
                        counter = counter + 1
                if counter > 0:  # Clear
                    ovr_l.append(1)
                    col_list.append(sample_n[-1].tolist())
                else:  # Tinted
                    ovr_l.append(0)
                    colour = col_proc(sample_n)
                    col_list.append(colour)
        dat = []
        dat.append(col_list)
        dat.append(ovr_l)
        with open(full_dir+'eyec_data.pickle', 'wb') as f1:
            pickle.dump(dat, f1)

    with open(full_dir+'eyec_data.pickle', 'rb') as f2:
        dat = pickle.load(f2)
        all_index = dat[-1]
        list_col = dat[:-1][0]

    return y_name, y_eye, y_shp, all_index, list_col
