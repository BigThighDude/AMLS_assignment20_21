import numpy as np
import cv2

# src = './celeba/img/'+str(14)+'.png'
src = '4.png'
img = cv2.imread(src,1)




print(img[271][204])
cv2.imshow('Image',img)
cv2.waitKey(0)


# 204,271 - location of eye colour
