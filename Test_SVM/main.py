import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from random import randint

no_pics = 100
hwp = 50
test_size = 20
hwpt = 10

x_tr = [[0]*3]*no_pics
y_tr = [0]*no_pics
x_ts = [[0]*3]*test_size

for i in range(0, hwp):
    x_tr[i] = i, randint(0,50), randint(0,50)
    y_tr[i] = 0

for i in range(hwp, no_pics):
    x_tr[i] = i, randint(50,100), randint(50,100)
    y_tr[i] = 1

for i in range(0, hwpt):
    x_ts[i] = i, randint(0,50), randint(0,50)

for i in range(hwpt, test_size):
    x_ts[i] = i, randint(50,100), randint(50,100)

x_tr = np.array(x_tr)
y_tr = np.array(y_tr)

clf = svm.SVC(gamma=0.01,C=1)
clf.fit(x_tr, y_tr)
print(x_ts)
print(clf.predict(x_ts))