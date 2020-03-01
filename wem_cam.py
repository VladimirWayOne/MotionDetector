import cv2
import imutils
from numba import jit
import numpy as np


def combine_boxes(box: (int, int, int, int), x: int, y: int, w: int, h: int):
    if ((x >= box[0]) and (x <= (box[0] + box[2]))):
        if ((y >= box[1]) and (y <= (box[1] + box[3]))):
            if (x + w) < (box[0] + box[2]):
                w = box[2]
            else:
                w = x + w - box[0]
            if (y + h) < (y + box[3]):
                h = box[3]
            else:
                h = y + h - box[1]
            x, y = box[0], box[1]
        elif ((x >= box[0]) and (x <= (box[0] + box[2]))) and ((y + h >= box[1]) and (y + h <= (box[1] + box[3]))):
            if (x + w) < (x + box[2]):
                w = box[2]
            else:
                w = x + w - box[0]
            h = box[1] + box[3] - y
            x = box[0]
        elif ((x + w >= box[0]) and (x + w <= box[0] + box[2])) and ((y + h >= box[1]) and (y + h <= (box[1] + box[3]))):
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
        if (x + w) < (x + box[2]):
            w = box[2]
        else:
            w = x + w - box[0]
        h = box[1] + box[3] - y
        x = box[0]
    elif ((x + w >= box[0]) and (x + w <= box[0] + box[2])) and ((y + h >= box[1]) and (y + h <= (box[1] + box[3]))):
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

capture = cv2.VideoCapture(0)    # 'http://root:root@192.168.0.101/mjpg/video.mjpg'

first_frame = None
bg_frames = []
box_buf = []
while True:

    ret, frame = capture.read()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(cv2.resize(gray, (920, 540), interpolation=cv2.INTER_CUBIC),
                            (21, 21), 0)
    frame = cv2. resize(frame, (920, 540))
    if first_frame is None:
        first_frame = gray
        bg_frames.append(first_frame)
        continue
    if len(bg_frames) < 10:
        bg_frames.insert(0, gray)
    else:
        bg_frames.pop(9)
        bg_frames.insert(0, gray)
    if len(bg_frames) < 10:
        frames_diff = cv2.absdiff(first_frame, gray)
    else:
        frames_diff = cv2.absdiff(bg_frames[5], gray)
    thresh_frame = cv2.threshold(frames_diff, 7, 240, cv2.THRESH_BINARY)[1]

    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for i in contours:
        if cv2.contourArea(i) < 8000:
            continue

        (x, y, w, h) = cv2.boundingRect(i)
       # print(x, y, w, h)

        if len(box_buf) < 3:
            box_buf.insert(0, (x, y, w, h))
        else:
            box_buf.pop(2)
            box_buf.insert(0, (x, y, w, h))

        for box in box_buf:
            (x, y, w, h) = combine_boxes(box, x, y, w, h)

        cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 255, 0), 3)

    cv2.imshow('orig', frame)
    #cv2.imshow('diff', frames_diff)
    cv2.imshow('frame', thresh_frame)
    # cv2.imshow('cont', cv2.drawContours(thresh_frame, contours, -1, (10,255,10), 5))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()
