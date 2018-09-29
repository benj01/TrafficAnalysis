from PIL import Image
from tracklet import BoundingBox


# switches axes and computes width and height
def yolo_box_to_bbox(box, padding):
    box = [int(x) for x in box]
    return (box[1] - padding, box[0]- padding,box[3] - box[1] + 2*padding, box[2] - box[0]+ 2*padding)

class DetectionProvider:
    def __init__(self, yolo, min_confidence = 0.6,  classes = ["car", "truck", "bus", "motorbike"], inflate = 0):
        self.yolo = yolo
        self.min_confidence = min_confidence
        self.classes = classes
        self.inflate = inflate
        
    def detect_boxes(self, frame, n):
        boxes = [ 
            BoundingBox(yolo_box_to_bbox(x[2], self.inflate), n, x[0], x[1])
            for x in self.yolo.detect_image(Image.fromarray(frame)) 
            if x[1] >= self.min_confidence and x[0] in self.classes
        ] 

    
        boxes.sort(key=lambda x: x.confidence, reverse=True)
        duplicates = []
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if boxes[i].iou(boxes[j]) >= 0.2:
                    duplicates.append(boxes[j])

        for duplicate in duplicates:
            if duplicate in boxes:
                boxes.remove(duplicate)

        return boxes

            
