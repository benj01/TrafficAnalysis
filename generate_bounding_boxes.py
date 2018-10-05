from optparse import OptionParser
from tqdm import tqdm
from detectionprovider import DetectionProvider
from tracklet import BoundingBox
from video import VideoStreamReader
from yolo import YOLO
import pandas as pd

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input")
    parser.add_option("-o", "--output", dest="output")
    parser.add_option("-p", "--padding", dest="padding")
    parser.add_option("-d", "--downscale", dest="downscale")
    parser.add_option("-s", "--skip", dest="skip")
    parser.add_option("-c", "--count", dest="count")
    parser.add_option("-f", "--fps_update", dest="fps_update")
    parser.add_option("-z", "--score", dest="min_score")
    
    (options, args) = parser.parse_args()
    print(options, args)
    padding, downscale = int(options.padding), int(options.downscale)
    reader = VideoStreamReader(options.input, seconds_count=int(options.count), seconds_skip=int(options.skip),downscale=downscale)

    yolo = YOLO(score=float(options.min_score))
    fps_update = int(options.fps_update)
    detection_provider = DetectionProvider(yolo, inflate=0)

    old_tracklets, new_tracklets = [], []
    frame_numbers = []
    xs = []
    ys = []
    ws = []
    hs = []
    object_classes = []
    confidences = []
    pbar = tqdm(total=reader.frame_count - reader.frame_skip)
    frame = reader.next_frame()
    while frame is not None:
        pbar.update()
        # detection step
        if reader.frame_no % fps_update == 0:
            detections = detection_provider.detect_boxes(frame, reader.frame_no)
            frame_bbox = BoundingBox.from_frame(frame, padding // downscale)
            # detect only inside frame_bbox region
            # could be multiple regions suitable for detecting objects
            detected_boxes = [x for x in detections if  x.is_inside(frame_bbox)]
            for b in detected_boxes:
                frame_numbers.append(b.frame_no)
                xs.append(b.dims[0])
                ys.append(b.dims[1])
                ws.append(b.dims[2])
                hs.append(b.dims[3])
                object_classes.append(b.object_class)
                confidences.append(b.confidence)
        frame = reader.next_frame()

    pd.DataFrame({
        "n" : frame_numbers,
        "x" : xs,
        "y" : ys,
        "w" : ws,
        "h" : hs,
        "object_classes" : object_classes,
        "confidence" : confidences
    }).to_csv(options.output)

    pbar.close()
    reader.release()