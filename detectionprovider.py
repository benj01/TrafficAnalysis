from PIL import Image

from utils import *

class DetectionProvider:
    def __init__(self, yolo, min_confidence = 0.6,  classes = ["car", "truck", "bus", "motorbike"], padding = 5):
        self.yolo = yolo
        self.min_confidence = min_confidence
        self.classes = classes
        self.padding = padding
        
    def detect_boxes(self, frame, n):
        boxes = [ 
            { "class": x[0], "confidence": x[1], "bbox":  yolo_box_to_bbox(x[2], self.padding), "n": n}
            for x in self.yolo.detect_image(Image.fromarray(frame)) 
            if x[1] >= self.min_confidence and x[0] in self.classes
        ] 

        boxes.sort(key=lambda x: x["confidence"], reverse=True)
        duplicates = []
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if iou(boxes[i]["bbox"], boxes[j]["bbox"]) >= 0.2:
                    duplicates.append(boxes[j])

        for duplicate in duplicates:
            if duplicate in boxes:
                boxes.remove(duplicate)

        return boxes

            
