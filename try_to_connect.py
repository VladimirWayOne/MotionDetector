import cv2
import numpy as np
import imutils

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 24)
first_frame = None


while True:
    ret, current_frame = capture.read()
    if current_frame is None:
        break
    current_frame = cv2.resize(current_frame, (int(current_frame.shape[1]*0.7), int(current_frame.shape[0]*0.7)),
                               interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 5, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    circles = cv2.HoughCircles(closed.copy(), cv2.HOUGH_GRADIENT, 1, closed.shape[0]/8, param1=100, param2=30,
                           minRadius=30, maxRadius=1000)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            # cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(current_frame, center, radius, (255, 0, 255), 3)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 1000:
            cv2.drawContours(current_frame, [approx], -1, (0, 255, 0), 4)
       # if len(approx) > 4:
       #     ellipse = cv2.fitEllipse(c)
       #     cv2.ellipse(current_frame, ellipse, (255, 0, 0), 2)

    cv2.imshow('orig', current_frame)
    cv2.imshow('edged', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
