import numpy as np
import cv2

src = './cartoon_set/img/'+str(3)+'.png'
img = cv2.imread(src,1)

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        # cv2.imshow('Image', img)

        # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        # cv2.imshow('Image', img)

# print(img[269][220])
cv2.imshow('Image',img)
cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)

# 204,271 - location of eye colour
# 220,269 - location of eye white
# 187,279 - location of glasses over skin ## probably dont need
# 249,236 - location of skin colour

############# key points ####################
# images are 500x500 pixels

############ face edges ##############
# 250,356 - lower lip
# 250,391 - under lowest chin

# 160,365 - pos3 - 1st
# 250,365 - pos3 - 2nd

# 215,335 - pos2 - 1st
# 130,335 - pos2 - 2nd

# 157,290 - pos1 - 1st
# 173,290 - pos1 - 2nd


