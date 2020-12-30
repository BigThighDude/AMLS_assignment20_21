import Utilities
# import Gen_model
import os
import pickle
from sklearn import decomposition
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from csv import reader
from sklearn import tree
import time


directory = './celeba'
model_src = './trained_gen/'
src_csv = directory+'/labels.csv'
file_list = os.listdir(model_src)
no_trained = int(len(file_list)/2)
model_nm = [0]*no_trained
array_nm = [0]*no_trained
model_str = [0]*no_trained
array_str = [0]*no_trained

for i in range(0, no_trained):
    x = i*2
    model_nm[i] = file_list[x]
    array_nm[i] = file_list[x+1]

    with open(model_src + model_nm[i], 'rb') as f1:  # open the pickle file containing the landmark data
        model_str[i] = pickle.load(f1)  # load the file into a temporary variable
    with open(model_src + array_nm[i], 'rb') as f2:  # open the pickle file containing the landmark data
        array_str[i] = pickle.load(f2)  # load the file into a temporary variable

face_list, points = Utilities.unpickle(directory)
use_pics = len(face_list)
points_src = './celeba/datapoints.pickle'
no_ratios = int(len(array_str[0])/2)



face_list, points = Utilities.unpickle(directory)
use_pics = len(face_list)
ratios = np.zeros([use_pics, no_ratios], dtype=float)
picked_lm = array_str[0]
ratios = Utilities.lm_to_points(face_list, points, picked_lm, ratios)
y_name, y_gen, y_smile = Utilities.csv_in(src_csv)
y_use = y_smile
x_train, y_train, x_test, y_exp = Utilities.data_split(0.75, use_pics, face_list, ratios, y_use)

################################ RFE ################################

start2 = time.time()
no_sel = 1
no_wtd = 24
rfe = RFE(estimator=tree.DecisionTreeClassifier(), n_features_to_select=no_sel)
rfe.fit(x_train, y_train)
sel_ft = rfe.support_
ranks = rfe.ranking_
ranks = ranks.tolist()

picked_lm2 = []
for i in range(0, no_wtd):
    idx = ranks.index(i+1)*2
    picked_lm2.append(picked_lm[idx])
    picked_lm2.append(picked_lm[idx+1])
end2 = time.time()
print(end2-start2)
print(ranks)
print(len(ranks))
print(picked_lm)
print(picked_lm2)


################################ PCA ##############################
# pca_no = 15
#
# pca = decomposition.PCA(n_components=pca_no)
# pc = pca.fit_transform(ratios)
# ratios = ratios.tolist()
# pc = pc.tolist()
# print(ratios[0])
# print(pc[0])
#
# print(pc[0])
# print(ratios[0])
# print(cont)
# print(vals)
# contribution = np.array(pca.explained_variance_ratio_)
# contribution = np.cumsum(contribution)
# xarr = list(range(0, len(contribution)))
#
# print(ratios[0])
#
# x_pos = [i for i, _ in enumerate(xarr)]
# plt.bar(x_pos, contribution)
# plt.show()


