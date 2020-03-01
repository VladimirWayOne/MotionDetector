import functions as fn
import cv2
import imutils

capture = cv2.VideoCapture(0)   # capture video from webcam
frame_cut = None
first_frame = None
bg_frames = []          # last frames buffer 
bg_frames_count = 6     # maximum frames in buffer
frame_num = True        # this is used to skip frames
while True:
    if frame_num is False:
        frame_num = True
    else:
        frame_num = False
        continue
    current_frame = capture.read()[1]
    if current_frame is None:
        break
    current_frame = fn.frame_resize(current_frame)
    output_frame = current_frame.copy()
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = fn.blur(gray, 7, 7)

    if len(bg_frames) < bg_frames_count:
        bg_frames.insert(0, gray)
        continue
    else:
        bg_frames.pop(bg_frames_count-1)
        bg_frames.insert(0, gray)

    frame_diff = cv2.absdiff(gray, bg_frames[len(bg_frames)-1])

    thresh_frame = cv2.threshold(frame_diff, 15, 200, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=1)
    contours = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for cnt in contours:
        if cv2.contourArea(cnt) < 8000:
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        w_cut, h_cut = w, h
        if w_cut < 300:
            w_cut = 300
        if h_cut < 300:
            h_cut = 300
        frame_cut = fn.moving_box(current_frame[y:y + h_cut, x:x + w_cut])
        cv2.rectangle(output_frame, (x, y), (w + x, h + y), (0, 255, 0), 3)

    for (x, y, w, h) in fn.face_detect(output_frame):
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('detector', output_frame)
    if frame_cut is not None:
        cv2.imshow('cut', frame_cut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
