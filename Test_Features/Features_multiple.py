import numpy as np
import cv2
import dlib
import time

no_pics = 100
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
total_nof = 0
noface = []
xlist = []

start = time.time()
for i in range(0,no_pics):
    xlist.append(i)
    src = './celeba/img/'+str(i)+'.jpg'
    img = cv2.imread(src, 1)
    # cv2.imshow('Image',img)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    # print(faces)

    x = list(faces)
    if not x:
        total_nof = total_nof+1
        noface.append(i)
        del xlist[-1]

end = time.time()

clean_pics = len(xlist)

print('number of noface: ', total_nof)
print('time to calculate: ', end-start)
# print(noface)
# print(xlist)









# for i in faces:
#     x1 = i.left()
#     y1 = i.top()
#     x2 = i.right()
#     y2 = i.bottom()
#     landmarks = predictor(gray, i)
#     for n in range(0, 68):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
#         cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
#
#
#
# cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
# cv2.imshow('Image',img)
# cv2.waitKey(0)
