from _collections import deque
import cv2
import numpy as np


def gen_movement_frame(frames: list, shape):
    acc = np.zeros(shape, dtype='float32')
    i = 0
    for f in frames:
        i += 1
        acc += f * i
    acc = acc / ((1 + i) / 2 * i)
    acc[acc > 254] = 255
    return acc


class MotionDetector:
    def __init__(self,
                 bg_subs_scale_percent=0.25,
                 bg_history=15,
                 movement_frames_history=5,
                 brightness_discard_level=20,
                 pixel_compression_ratio=0.1,
                 group_boxes=True,
                 expansion_step=1):

        self.bg_subs_scale_percent = bg_subs_scale_percent
        self.bg_history = bg_history
        self.group_boxes = group_boxes
        self.expansion_step = expansion_step
        self.movement_fps_history = movement_frames_history
        self.brightness_discard_level = brightness_discard_level
        self.pixel_compression_ratio = pixel_compression_ratio

        self.bg_frames = deque(maxlen=bg_history)
        self.movement_frames = deque(maxlen=movement_frames_history)
        self.count = 0
        self.background_acc = None
        self.background_frame = None
        self.boxes = None
        self.movement = None
        self.color_movement = None
        self.gs_movement = None
        self.detection = None
        self.detection_boxed = None
        self.frame = None

    @classmethod
    def blur_resize(cls, frame, width, height):
        return cv2.GaussianBlur(cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC), (5, 5), 0)

    def _update_background(self, frame_fp32):
        if self.movement_frames.maxlen == len(self.movement_frames):
            current_frame = self.movement_frames[0]
        else:
            current_frame = frame_fp32
        
        if self.background_acc is None:
            self.background_acc = frame_fp32
        else:
            self.background_acc = self.background_acc + current_frame

            if self.bg_frames.maxlen == len(self.bg_frames):
                subs_frame = self.bg_frames[0]
                self.background_acc = self.background_acc - subs_frame

            self.bg_frames.append(current_frame)

    def __detect_movement(self, frame_fp32):
        self.movement_frames.append(frame_fp32)
        movement_frame = gen_movement_frame(list(self.movement_frames), frame_fp32.shape)
        self.background_frame = self.background_acc / len(self.bg_frames)
        self.background_frame[self.background_frame > 254] = 255
        if len(self.bg_frames):
            movement = cv2.absdiff(movement_frame, self.background_frame)
        else:
            movement = np.zeros(movement_frame.shape)
        self.color_movement = movement
        movement[movement < self.brightness_discard_level] = 0
        movement[movement > 0] = 254
        movement = movement.astype('uint8')
        movement = cv2.cvtColor(movement, cv2.COLOR_BGR2GRAY)
        movement[movement > 0] = 254
        return movement
