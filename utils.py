import cv2

def show_bbox(bbox, frame,color,width=2):
    p1 = (int(bbox[0]),int(bbox[1]))
    p2 = (int(bbox[2]) + p1[0],int(bbox[3])+ p1[1])
    cv2.rectangle(frame, p1, p2, color, width)


def box_intersection(a, b):
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[0] + a[2], b[0] + b[2])
    y_bottom = min(a[1] + a[3], b[1] + b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)

def box_iou(a, b):
    intersection_area = box_intersection(a, b)
    a_area = a[2] * a[3]
    b_area = b[2] * b[3]
    
    return intersection_area / float(a_area + b_area - intersection_area)

def box_overlap(a, b):
    intersection_area = box_intersection(a, b)
    a_area = a[2] * a[3]
    b_area = b[2] * b[3]
    a_fraction = intersection_area / float(a_area)
    b_fraction = intersection_area / float(b_area)

    return max(a_fraction,b_fraction)

def inside_frame(bbox, shape):
    return bbox[0] >= 0 and bbox[1] >= 0 and bbox[0] + bbox[2] <= shape[1] and bbox[1] + bbox[3] <= shape[0]

# switches axes and computes width and height
def yolo_box_to_bbox(box, padding):
    box = [int(x) for x in box]
    return (box[1] - padding, box[0]- padding,box[3] - box[1] + 2*padding, box[2] - box[0]+ 2*padding)