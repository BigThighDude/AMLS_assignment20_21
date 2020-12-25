import numpy as np
import cv2
import dlib
import time
import pickle
import utilities

start1 = time.time()
no_pics = 5000
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
total_nof = 0
noface = []
xlist = []
points = np.array([[[None]*2]*68]*no_pics)
end1 = time.time()

start2 = time.time()
for i in range(0,no_pics):

    # Open image and find face
    src = './celeba/img/'+str(i)+'.jpg'
    img = cv2.imread(src, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if not list(faces):
        total_nof = total_nof+1
    else:
        xlist.append(i)
        face = list(faces)[0]
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            points[i][n][0] = landmarks.part(n).x
            points[i][n][1] = landmarks.part(n).y

end2 = time.time()
print(end1-start1)
print(end2-start2)
print(len(points))

# pickling points data
with open('datapoints.pickle', 'wb') as f:
    pickle.dump(points, f)

# cv2.imshow('Image',img)
# cv2.waitKey(0)












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
