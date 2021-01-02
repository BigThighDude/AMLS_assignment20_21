import numpy as np
import cv2
from csv import reader
from matplotlib import pyplot as plt
from scipy import stats
from collections import Counter
from sklearn import svm


directory = './cartoon_set/'
csv_src = directory+'labels.csv'
img_src = directory+'img/'

with open(csv_src) as file:
    labels = list(reader(file, delimiter='\t'))
    labels = labels[1:]

no_pics = len(labels)
y_name = []
y_eyec = []

for i in range(no_pics):
    y_name.append(labels[i][3])
    y_eyec.append(int(labels[i][1]))

cols = np.zeros(shape=(no_pics,2,3))

picno = 34
max_fn = 100
white = np.array([255,255,255])
black = np.array([0,0,0])
ovr_l = []
list_w = []
list_t = []
list_o = []
col_list = []
tint_list = []

for i in range(0, max_fn):
    img = cv2.imread(img_src + y_name[i])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sample_n = img[:][263]
    sample_n = sample_n[190:209]
    mpi = sample_n[int(len(sample_n)/2)]
    if np.allclose(mpi,sample_n):       # Obscured
        list_o.append(i)
        ovr_l.append(-1)
    elif len(np.where(sample_n>0.9*white)[0])>0:   # White
        list_w.append(i)
        ovr_l.append(1)
        w_find = np.unique(np.where(sample_n == white)[0])
        b_find = np.append(w_find, np.unique(np.where(sample_n == black)[0]))
        cleaned = np.delete(sample_n, b_find, 0)
        colour = np.array(stats.mode(cleaned))
        col_list.append(colour[0][0])
        # print(colour[0][0])
    else:                               # Tinted
        list_t.append(i)
        ovr_l.append(0)
        print(sample_n)
        # print(i)
    rgb_array = sample_n
    img2 = np.array(rgb_array, dtype=int).reshape((1, len(rgb_array), 3))
    plt.imshow(img2, extent=[0, 16000, 0, 1], aspect='auto')
    plt.show()
use_no = len(col_list)
print(use_no)

y_use = []
for i in range(0, use_no):
    idx = list_w[i]
    y_use.append(y_eyec[idx])

# use_no = len(list_t)
# print(use_no)
# print(list_t)
# for i in range(0, use_no):
#     idx = list_t[i]
#     y_use.append(y_eyec[idx])

###############SPLIT DATA - Tint ###################
# split = int(0.75*use_no)
# x_train = tint_list[:split]
# y_train = y_use[:split]
# x_test = tint_list[split:]
# y_test = y_use[split:]

############### SPLIT DATA - White #################
# split = int(0.75*use_no)
# x_train = col_list[:split]
# y_train = y_use[:split]
# x_test = col_list[split:]
# y_test = y_use[split:]

################ SVM ##################
# clf = svm.SVC(kernel='linear')
# clf.fit(x_train, y_train)
# accuracy = clf.score(x_test, y_test)
# print(accuracy)

################ Print Info ################
# print("No. eye_whites: ", len(list_w))
# print("No. tint: ", len(list_t))
# print("No. obs: ", len(list_o))
# print(ovr_l)
# print(sample_n,"\n")

############### Plot Slice ####################
# rgb_array = sample_n
# img2 = np.array(rgb_array, dtype=int).reshape((1, len(rgb_array), 3))
# plt.imshow(img2, extent=[0, 16000, 0, 1], aspect='auto')
# plt.show()

############### RGB values for each eye colour ############################
# 0 - Brown - 124, 72, 53       RED
# 1 - Blue - 52, 114, 160       BLUE
# 2 - Green - 114, 147, 67      GREEN
# 3 - Grey - 148, 164, 160      SAME MID
# 4 - Black - 0, 0, 0           SAME LOW

################ Key Locations #######################
# Slice coordinates: 189,263:215,263

############# Data info #################
# No. eye whites:       7790 - 100% accurate
# No. tinted:           352
# No. obscured:         1858

