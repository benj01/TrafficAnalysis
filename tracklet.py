import cv2
import numpy as np
from random import randint

from utils import *

class Tracklet:
    def __init__(self, box, frame, min_iou=0.2, tracker_factory = cv2.TrackerCSRT_create):
        self.min_iou = min_iou
        self.tracker_factory = tracker_factory
        self.tracker = self.tracker_factory()
        self.ended =  not self.tracker.init(frame, box["bbox"])
        self.boxes = [ box ]
        self.color = (randint(0,255),randint(0,255),randint(0,255))
    
    def propagate(self, frame, n):
        ok, bbox = self.tracker.update(frame)

        # check if we are inside image
        if ok and inside_frame(bbox, frame.shape):
            self.boxes.append( { "bbox" : bbox , "n" : n })
        else:
            self.ended = True
            
    def show(self, frame,width=2):
        bbox = self.boxes[-1]["bbox"]
        show_bbox(bbox, frame, self.color,width)
    
    # returns true if tracklet was updated by one of newly discovered boxes
    def update(self, frame, boxes):
        last_box = self.boxes[-1]
        overlaps = np.array([ box_iou(last_box["bbox"], x["bbox"]) if inside_frame(x["bbox"], frame.shape) else 0.0 for x in boxes])
        overlapping = overlaps > self.min_iou
        
        if np.sum(overlapping) > 0:
            i = np.argmax(overlaps)
            new_box = boxes[i]
            self.boxes.append(new_box)
            self.tracker = self.tracker_factory()
            self.ended =  not self.tracker.init(frame, new_box["bbox"])
            return list(np.flatnonzero(overlapping))
        else:
            return []