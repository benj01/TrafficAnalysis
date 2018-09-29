import sys
import cv2
from optparse import OptionParser

from detectionprovider import DetectionProvider
from tracklet import Tracklet, BoundingBox, create_tracklets, update_tracklets
from video import VideoStreamReader, VideoStreamWriter
from yolo import YOLO

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input")
    parser.add_option("-o", "--output", dest="output")
    parser.add_option("-p", "--padding", dest="padding")
    parser.add_option("-d", "--downscale", dest="downscale")
    parser.add_option("-s", "--skip", dest="skip")
    parser.add_option("-c", "--count", dest="count")
    
    (options, args) = parser.parse_args()
    print(options, args)
    padding, downscale = int(options.padding), int(options.downscale)
    reader = VideoStreamReader(options.input, seconds_count=int(options.count), seconds_skip=int(options.skip),downscale=downscale)
    writer = VideoStreamWriter(options.output, width=reader.width,height=reader.height,fps=reader.fps)

    yolo = YOLO()
    yolo_update = 0.5 # detection update in seconds
    fps_update = int(yolo_update * reader.fps)
    detection_provider = DetectionProvider(yolo, min_confidence=0.6,inflate=3)

    old_tracklets, new_tracklets = [], []

    pbar = tqdm(total=reader.frame_count - reader.frame_skip)
    frame = reader.next_frame()
    while frame is not None:
        pbar.update(reader.frame_no - reader.frame_skip)
        # detection step
        if reader.frame_no % fps_update == 0:
            completed = 100 * (reader.frame_no - reader.frame_skip) // reader.frame_count
            print( (completed*"=" ) + ((100-completed)*"." ) ) 
            detections = detection_provider.detect_boxes(frame, reader.frame_no)
            frame_bbox = BoundingBox.from_frame(frame, padding // downscale)
            # detect only inside frame_bbox region
            # could be multiple regions suitable for detecting objects
            detected_boxes = [x for x in detections if  x.is_inside(frame_bbox)]
            old_tracklets += new_tracklets
            new_tracklets = create_tracklets(frame, detected_boxes) 
        
        # propagation step
        update_tracklets(old_tracklets, frame, reader.frame_no)  
        update_tracklets(new_tracklets, frame, reader.frame_no)
        
        # find boxes with high iou indicating that detectors may track the same item
        pairs = []
        for i, old_tracklet in enumerate(old_tracklets):
            for j, new_tracklet in enumerate(new_tracklets):
                if old_tracklet.last_box().iou(new_tracklet.last_box()) > 0.3:
                    new_tracklet.merge(old_tracklet)
                    pairs.append( (i, j) )

        # removing old
        for i in sorted(list(set([x[0] for x in pairs])),reverse=True):
            old_tracklets.pop(i)
        
        # checking for exiting bboxes if their center is outside 
        old_ids = [ i for i, old_tracklet in enumerate(old_tracklets) if not old_tracklet.last_box().is_center_inside(frame_bbox)]

        for i in sorted(old_ids,reverse=True):
            old_tracklets.pop(i)
            
        new_ids = [ i for i, new_tracklet in enumerate(new_tracklets) if not new_tracklet.last_box().is_center_inside(frame_bbox)]
        for i in sorted(new_ids,reverse=True):
            new_tracklets.pop(i)
        
        for old_tracklet in old_tracklets:
            old_tracklet.show_history(frame)
            
        for new_tracklet in new_tracklets:
            new_tracklet.show_history(frame)

        writer.write(frame)
        frame = reader.next_frame()

    pbar.close()
    reader.release()
    writer.release()