import cv2
import numpy as np
import imutils


img = cv2.imread('/home/vladimir/Изображения/test_frame.png')
#img = cv2.resize(img, (int(img.shape[1]*0.8), int(img.shape[0]*0.8)), interpolation=cv2.INTER_CUBIC)
img = img[153:262, 511:590]
empty_image = img.copy()
empty_image[:, :] = 255
img = cv2.GaussianBlur(img, (5, 5), 0)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(grey, 50, 250)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#closed = cv2.dilate(closed, None, iterations=1)
#cv2.imshow('edged', closed)
cv2.imshow('img1', img)
cnts = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

circles = cv2.HoughCircles(closed.copy(), cv2.HOUGH_GRADIENT, 1, closed.shape[0]/8, param1=100, param2=30,
                           minRadius=10, maxRadius=100)

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05*peri, True)
    if len(approx) == 4:
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)




#thresh = cv2.threshold(grey, 127, 255, 0)[1]

#print(len(thresh[0])*len(thresh))
cv2.imshow('cnt', cv2.drawContours(empty_image, cnts, -1, (0, 0, 0), 1))
#cv2.imshow('image', thresh)
cv2.imshow('img', img)
cv2.imshow('closed', closed)
cv2.waitKey(0)
#196416
#3147264