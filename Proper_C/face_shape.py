from csv import reader
import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import path
import pickle
import time
from sklearn import svm

directory = './cartoon_set/'
csv_src = directory+'labels.csv'
img_src = directory+'img/'

with open(csv_src) as file:
    labels = list(reader(file, delimiter='\t'))
    labels = labels[1:]

no_pics = len(labels)
y_name = []
y_fshp = []
for i in range(no_pics):
    y_name.append(labels[i][3])
    y_fshp.append(int(labels[i][2]))

if not path.isfile(directory+'datapoints.pickle'):  # if the data file containing facial landmark info doesnt exist
    slice_points = [[[160,365],[250,365]],[[130,335],[215,335]],[[157,290],[173,290]]]
    no_slice = len(slice_points)
    points = np.zeros(shape=(no_slice,no_pics,1))
    print("Acquiring points\n")
    start = time.time()
    for m in range(0, no_pics):
        for n in range(0, no_slice):
            img = cv2.imread(img_src + y_name[m])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            slice_d = slice_points[n][0][1]
            slice_s = slice_points[n][0][0]
            slice_e = slice_points[n][1][0]
            sample_n = img[:][slice_d]
            sample_n = sample_n[slice_s:slice_e]
            avg = np.zeros(shape=(len(sample_n),1), dtype=float)
            for i in range(0, len(sample_n)):
                avg[i] = np.mean(sample_n[i])
            min_val = avg.min()
            pos_list = np.where(avg==min_val)
            pos_mid = pos_list[0][int(len(pos_list[0])/2)]
            avg_at_pos = avg[pos_mid]
            if avg_at_pos<35:
                points[n][m] = int(pos_mid)
            else:
                if n==0:
                    points[n][m]=53.4
                elif n==1:
                    points[n][m]=52.0
                elif n==2:
                    points[n][m]=9.2
                # points[n][m] = int(len(sample_n)*0.5)
    end = time.time()
    print("Time to acquire points: ",end-start)
    with open(directory+'datapoints.pickle', 'wb') as f1:
        pickle.dump(points, f1)

with open(directory+'datapoints.pickle', 'rb') as f2:
    points = pickle.load(f2)

points_sort = []
for i in range(0, no_pics):
    temp = []
    temp.append(points[0][i][0])
    temp.append(points[1][i][0])
    temp.append(points[2][i][0])
    points_sort.append(temp)

split = int(no_pics*0.75)
x_train = points_sort[:split]
y_train = y_fshp[:split]
x_test = points_sort[split:]
y_test = y_fshp[split:]

# print(points_sort[0])

print("Training SVM")
start = time.time()
clf = svm.SVC(kernel='linear')   # poly4/5/6: 97.3%, poly3: 97.36%##fastest with wp, linear: 97.4%##fastest with w
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
end = time.time()
print("Time to train: ",end-start)
print("Accuracy: ",accuracy)


############ shows sample slice #################
# rgb_array = sample_n
# img = np.array(rgb_array, dtype=int).reshape((1, len(rgb_array), 3))
# plt.imshow(img, extent=[0, 16000, 0, 1], aspect='auto')
# plt.show()


############ face edges ##############
# 250,356 - lower lip
# 250,391 - under lowest chin

# 160,365 - pos3 - 1st
# 250,365 - pos3 - 2nd

# 130,335 - pos2 - 1st
# 215,335 - pos2 - 2nd

# 157,290 - pos1 - 1st
# 173,290 - pos1 - 2nd
