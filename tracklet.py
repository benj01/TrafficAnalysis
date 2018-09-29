import cv2
from random import randint

def random_color():
    return (randint(0,255),randint(0,255),randint(0,255))
            
def create_tracklets(frame, bboxes, tracker = cv2.TrackerCSRT_create):
    return [ Tracklet(frame, bbox,color=random_color()) for bbox in bboxes]

def update_tracklets(trackers, frame, frame_no):
    for t in trackers:
        t.propagate(frame, frame_no)

class BoundingBox:
    def __init__(self, dims, frame_no, object_class = None, confidence = None):
        self.dims = dims
        self.frame_no = frame_no
        self.object_class = object_class
        self.confidence = confidence

    def center(self):
        p1 = (int(self.dims[0]),int(self.dims[1]))
        return (int(self.dims[2] / 2)  + p1[0],int(self.dims[3]/ 2)+ p1[1])


    def show_center(self, frame, color,width=2):
        cv2.circle(frame, self.center(), 3, color, width)

    def show(self, frame, color,width=2):
        p1 = (int(self.dims[0]),int(self.dims[1]))
        p2 = (int(self.dims[2]) + p1[0],int(self.dims[3])+ p1[1])
        cv2.rectangle(frame, p1, p2, color, width)
        self.show_center(frame, color,width)


    def intersection(self, b):
        x_left = max(self.dims[0], b.dims[0])
        y_top = max(self.dims[1], b.dims[1])
        x_right = min(self.dims[0] + self.dims[2], b.dims[0] + b.dims[2])
        y_bottom = min(self.dims[1] + self.dims[3], b.dims[1] + b.dims[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        return (x_right - x_left) * (y_bottom - y_top)

    def iou(self, b):
        intersection_area = self.intersection(b)
        a_area = self.dims[2] * self.dims[3]
        b_area = b.dims[2] * b.dims[3]
        
        return intersection_area / float(a_area + b_area - intersection_area)

    def is_inside(self, b):
        return self.dims[0] >= b.dims[0] and \
            self.dims[1] >= b.dims[1] and \
            self.dims[0] + self.dims[2] <= b.dims[0] + b.dims[2] and \
            self.dims[1] + self.dims[3]  <= b.dims[1] + b.dims[3]

    
    def is_center_inside(self, b):
        x = self.center()
        return x[0] >= b.dims[0] and \
            x[1] >= b.dims[1] and \
            x[0] <= b.dims[0] + b.dims[2] and \
            x[1] <= b.dims[1] + b.dims[3]


    @staticmethod
    def from_frame(frame, padding = 0):
        dims = (padding,padding,frame.shape[1]- 2 * padding, frame.shape[0]-2 * padding)
        return BoundingBox(dims,0)


class Tracklet:
    def __init__(self, frame, bbox, color = (0, 255, 0),  tracker = cv2.TrackerCSRT_create):
        self.tracker = tracker()
        self.tracker.init(frame, bbox.dims)
        self.color = color
        self.bboxes = [ bbox ]

    def propagate(self, frame, frame_no):
        _, dims = self.tracker.update(frame)
        bbox = BoundingBox(dims, frame_no)
        self.bboxes.append(bbox)

    def merge(self, other):
        self.color = other.color
        self.bboxes = [ x for x in sorted(other.bboxes + self.bboxes, key=lambda x: x.frame_no)]


    def last_box(self):
        return self.bboxes[-1]

    
    def last_boxes(self, n):
        return self.bboxes[-n:]

    

    def show_history(self, frame, width = 2, n = 30):
        self.last_box().show(frame, self.color, width=width)

        for b in self.last_boxes(n):
            b.show_center(frame, self.color, width=width)

    

    