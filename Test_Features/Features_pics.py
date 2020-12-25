import numpy as np
import cv2
import dlib

src = './celeba/img/'+str(1)+'.jpg'
img = cv2.imread(src,1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

faces = detector(gray)

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    print("1")

    landmarks = predictor(gray, face)
    print("2")
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

print(list(faces)[0])
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imshow('Image',img)
cv2.waitKey(0)

# x = list(faces)
# print(x)
# if not x:
#     print('test')
