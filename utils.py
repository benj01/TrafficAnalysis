import cv2
from random import randint

def random_color():
    return (randint(0,255),randint(0,255),randint(0,255))

def avg_bbox(a,b):
    return ((a[0] + b[0])//2, (a[1] + b[1])//2,(a[2] + b[2])//2,(a[3] + b[3])//2)
                

def show_bbox(bbox, frame,color,width=2):
    p1 = (int(bbox[0]),int(bbox[1]))
    p2 = (int(bbox[2]) + p1[0],int(bbox[3])+ p1[1])
    cv2.rectangle(frame, p1, p2, color, width)
    cv2.circle(frame, center(bbox), 3, color, width)

def draw_gftt(box, frame, color):
    box_image = frame[int(box[1]):int(box[1]+box[3]),int(box[0]):int(box[0]+box[2])]
    if box_image.shape[0] * box_image.shape[1] > 0:
        gray= cv2.cvtColor(box_image.copy(),cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray,30,0.01,10)
        corners = np.int0(corners)

        for i in corners:
            x,y = i.ravel()
            cv2.circle(box_image,(x,y),2,color,-1)
            
def create_trackers(frame, bboxes, tracker = cv2.TrackerCSRT_create):
    trackers = []
    for bbox in bboxes:
        t = tracker()
        t.init(frame,bbox)
        trackers.append({"tracker" : t, "color" : (0, 255, 0), "boxes": [bbox]})
    return trackers

def update_trackers(trackers, frame):
    successes, bboxes = [], []
    for t in trackers:
        success,bbox = t["tracker" ].update(frame)
        t["boxes"].append(bbox)
        successes.append(success)
        bboxes.append(bbox)
        
    return successes, bboxes

def intersection(a, b):
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[0] + a[2], b[0] + b[2])
    y_bottom = min(a[1] + a[3], b[1] + b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)

def iou(a, b):
    intersection_area = intersection(a, b)
    a_area = a[2] * a[3]
    b_area = b[2] * b[3]
    
    return intersection_area / float(a_area + b_area - intersection_area)

def center(bbox):
    p1 = (int(bbox[0]),int(bbox[1]))
    return (int(bbox[2] / 2)  + p1[0],int(bbox[3]/ 2)+ p1[1])

def point_inside_frame(x, shape, padding):
    return x[0] >= padding and x[1] >= padding and x[0] + padding <= shape[1] and x[1] + padding <= shape[0]

def inside_frame(a, b):
    return a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3]  <= b[1] + b[3]

def bbox_from_frame(frame, padding):
    return (padding,padding,frame.shape[1]- 2 * padding, frame.shape[0]-2 * padding)

# switches axes and computes width and height
def yolo_box_to_bbox(box, padding):
    box = [int(x) for x in box]
    return (box[1] - padding, box[0]- padding,box[3] - box[1] + 2*padding, box[2] - box[0]+ 2*padding)