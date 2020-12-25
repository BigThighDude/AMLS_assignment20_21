import numpy as np
import cv2
import dlib
import pickle


no_pics = 10
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
total_nof = 0
noface = []
xlist = []
points = np.array([[[None]*2]*68]*no_pics)

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

# pickling points data
# with open('datapoints.pickle', 'wb') as f:
#     pickle.dump(points, f)