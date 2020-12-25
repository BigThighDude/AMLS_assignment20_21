import numpy as np
import cv2
import csv
from sklearn import svm

no_pics = 50
eye_c = [[None]*3]*no_pics
y_dat = [None]*no_pics
y_col = [None]*no_pics

for i in range(0,no_pics):
    src = './cartoon_set/img/'+str(i)+'.png'
    img = cv2.imread(src,1)
    eye_c[i] = list(img[271][204])

with open('./cartoon_set/labels.csv') as file:
    datread = csv.reader(file, delimiter='\t')
    temp = list(datread)
    temp = temp[1:]
    for i in range(0,no_pics):
        y_dat[i] = temp[i]

for i in range(0,no_pics):
    y_col[i] = y_dat[i][1]
y_col = np.array(y_col)

src = './cartoon_set/img/'+str(11)+'.png'
img = cv2.imread(src,1)
eye_test = [[None]*3]*1
eye_test[0] = list(img[271][204])
print(eye_test)

# print(eye_test)
# print(y_col)
# print(eye_c)

clf = svm.SVC(gamma=0.01,C=1)
# clf.fit(eye_c,y_col)
# print(clf.predict(eye_test))



# 271, 204 - location of eye colour
