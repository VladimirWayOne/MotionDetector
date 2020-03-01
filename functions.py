import cv2
import imutils


frame_scale_percent = 0.5
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def frame_resize(f):
    """ Resize frame """
    global frame_scale_percent
    return cv2.resize(f, (int(f.shape[1]*frame_scale_percent), int(f.shape[0]*frame_scale_percent)),
                      interpolation=cv2.INTER_CUBIC)


def combine_boxes(box: (int, int, int, int), x: int, y: int, w: int, h: int):
    """This function need for combine green rectangles"""
    if box[0] <= x <= box[0] + box[2]:
        if box[1] <= y <= box[1] + box[3]:
            if x + w < box[0] + box[2]:
                w = box[2]
            else:
                w = x + w - box[0]
            if (y + h) < (y + box[3]):
                h = box[3]
            else:
                h = y + h - box[1]
            x, y = box[0], box[1]
        elif ((x >= box[0]) and (x <= (box[0] + box[2]))) and ((y + h >= box[1]) and (y + h <= (box[1] + box[3]))):
            if x + w < x + box[2]:
                w = box[2]
            else:
                w = x + w - box[0]
            h = box[1] + box[3] - y
            x = box[0]
        elif (x + w >= box[0]) and (x + w <= box[0] + box[2]) and ((y + h >= box[1]) and (y + h <= (box[1] + box[3]))):
            w = box[0] + box[2] - x
            if y > box[1]:
                y = box[1]
                h = box[3]
            else:
                h = box[1] + box[3] - y
        elif ((x + w >= box[0]) and (x + w <= box[0] + box[2])) and ((y >= box[1]) and (y <= (box[1] + box[3]))):
            w = box[0] + box[2] - x
            h = x + h - box[1]
            y = box[1]
    elif ((x >= box[0]) and (x <= (box[0] + box[2]))) and ((y + h >= box[1]) and (y + h <= (box[1] + box[3]))):
        if w < box[2]:
            w = box[2]
        else:
            w = x + w - box[0]
        h = box[1] + box[3] - y
        x = box[0]
    elif (x + w >= box[0]) and (x + w <= box[0] + box[2]) and (y + h >= box[1]) and (y + h <= (box[1] + box[3])):
        w = box[0] + box[2] - x
        if y > box[1]:
            y = box[1]
            h = box[3]
        else:
            h = box[1] + box[3] - y
    elif ((x + w >= box[0]) and (x + w <= box[0] + box[2])) and ((y >= box[1]) and (y <= (box[1] + box[3]))):
        w = box[0] + box[2] - x
        h = x + h - box[1]
        y = box[1]

    return x, y, w, h


def blur(img, k1, k2):
    """Blur image"""
    return cv2.GaussianBlur(img, (k1, k2), 0)


def find_contours(img, dark, bright):
    """Find contour of moving object"""
    thresh_frame = cv2.threshold(img, dark, bright, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)
    contours = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)


def find_rectangle(cnt):
    peri = cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, 0.1 * peri, True)


def moving_box(img):
    """Find rectangles and ellipses on frame with moving object"""
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(edged, cv2.MORPH_RECT, kernel)

    contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.08 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

        #ellipse = cv2.fitEllipse(c)
        #cv2.ellipse(img, ellipse, (255, 0, 0), 2)
    return img


def face_detect(img):
    global face_cascade

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces
