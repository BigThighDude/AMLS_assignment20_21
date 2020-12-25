import numpy as np
import cv2


src = './cartoon_set/img/'+str(161)+'.png'
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

x = [[None]*3]

eyc = img[271][204]
eyw = img[269][220]
skc = img[236][249]


white = np.array([255, 255, 255])

x[0] = white - eyw
x[1] = eyc - x[0]
print(eyw)
print(eyc)
print(x)




# print(img[269][220])
cv2.imshow('Image',img)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)


# 204,271 - location of eye colour
# 220,269 - location of eye white
# 187,279 - location of glasses over skin ## probably dont need
# 249,236 - location of skin colour
