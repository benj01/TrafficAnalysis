from random import randint
import numpy as np
import cv2
import tensorflow as tf
from keras.backend import set_session

def random_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

def limit_gpu_memory(fraction):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))

class Detection:
    def __init__(self, tlwh, frame_no, object_class=None, confidence=None, predicted=False):
        self.tlwh = np.array(tlwh).astype(float)
        self.frame_no = frame_no
        self.object_class = object_class
        self.confidence = confidence
        self.predicted = predicted

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def show_center(self, frame, color ,width=2):
        xyah = self.to_xyah()
        cv2.circle(frame, (int(xyah[0]), int(xyah[1])), 3, color, width)

    def show(self, frame, color=(0,0,0),width=2):
        tlbr = self.to_tlbr()
        cv2.rectangle(frame, (int(tlbr[0]) ,int(tlbr[1])), (int(tlbr[2]) ,int(tlbr[3])), color, width)

    def intersection(self, x):
        a = self.to_tlbr()
        b = x.to_tlbr()
        x_left = max(a[0], b[0])
        y_top = max(a[1], b[1])
        x_right = min(a[2], b[2])
        y_bottom = min(a[3], b[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        return (x_right - x_left) * (y_bottom - y_top)

    def iou(self, b):
        intersection_area = self.intersection(b)
        a_area = self.tlwh[2] * self.tlwh[3]
        b_area = b.tlwh[2] * b.tlwh[3]
        return intersection_area / float(a_area + b_area - intersection_area)

    def is_inside(self, x):
        a, b = self.to_tlbr(), x.to_tlbr()
        return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3]  <= b[3]

    def is_center_inside(self, x):
        a, b = self.to_xyah(), x.to_tlbr()
        return a[0] >= b[0] and a[1] >= b[1] and a[0] <= b[2] and a[1] <= b[3]

    def get_ious(self, detections):
        return [self.iou(x) for x in detections]

    def get_max_iou(self, detections):
        ious = self.get_ious(detections)
        i = np.argmax(ious)
        return i, ious[i]

    @staticmethod
    def from_frame(shape, padding = 0):
        tlwh = (padding ,padding ,shape[0 ]- 2 * padding, shape[1 ] -2 * padding)
        return Detection(tlwh ,0)
