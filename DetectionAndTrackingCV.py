from optparse import OptionParser
from detectionprovider import DetectionProvider
from tracking import Detection, create_tracks, update_tracks, Track
from video import VideoStreamReader, VideoStreamWriter
from yolo import YOLO
from tqdm import tqdm
import tensorflow as tf
import cv2
from keras.backend.tensorflow_backend import set_session

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    set_session(tf.Session(config=config))


    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input")
    parser.add_option("-o", "--output", dest="output")
    parser.add_option("-p", "--padding", dest="padding")
    parser.add_option("--width", dest="width")
    parser.add_option("--height", dest="height")
    parser.add_option("-s", "--skip", dest="skip")
    parser.add_option("-c", "--count", dest="count")
    
    (options, args) = parser.parse_args()
    print(options, args)
    padding, width, height = int(options.padding), int(options.width), int(options.height)
    reader = VideoStreamReader(options.input, seconds_count=int(options.count), seconds_skip=int(options.skip),width=width,height=height)
    writer = VideoStreamWriter(options.output, width=reader.width,height=reader.height,fps=reader.fps)

    yolo = YOLO()
    min_iou = 0.3
    min_iou_to_discard = 0.2
    fps_update = 5
    detection_provider = DetectionProvider(yolo, inflate=3)

    old_tracks, new_tracks = [], []
    frame_bbox = Detection.from_frame((width,height), padding)

    pbar = tqdm(total=reader.frame_count - reader.frame_skip)
    while True:
        frame = reader.next_frame()
        if frame is None:
            break

        pbar.update()
        # detection step
        if reader.frame_no % fps_update == 0:
            detections = detection_provider.detect_boxes(frame, reader.frame_no)
            # detect only inside frame_bbox region
            # could be multiple regions suitable for detecting objects
            detected_boxes = [x for x in detections if x.is_inside(frame_bbox)]
            old_tracks += new_tracks
            new_tracks = create_tracks(frame, detected_boxes, tracker=cv2.TrackerMOSSE_create)
        
        # propagation step
        update_tracks(old_tracks, frame, reader.frame_no)
        update_tracks(new_tracks, frame, reader.frame_no)
        
        # find boxes with high iou indicating that detectors may track the same item
        matched_tracks = []
        old_track: Track
        for i, old_track in enumerate(old_tracks):
            if len(new_tracks) > 0:
                j, iou = old_track.get_max_iou(new_tracks)
                if iou > min_iou: #remove matched from further matching process
                    new_tracks[j].merge(old_track)
                    matched_tracks.append(new_tracks.pop(j))
                    old_tracks.remove(old_track)

        old_tracks += matched_tracks # add matched tracks to old

        for track in new_tracks:
            if len(old_tracks) > 0:
                _, iou = track.get_max_iou(old_tracks)
                if iou > min_iou_to_discard:  # remove unmatched with sufficient high iou
                    new_tracks.remove(track)

        # checking for exiting bboxes if their center is outside
        for track in old_tracks:
            if not track.last_box().is_center_inside(frame_bbox):
                old_tracks.remove(track)

        for track in new_tracks:
            if not track.last_box().is_center_inside(frame_bbox):
                new_tracks.remove(track)

        
        for old_tracklet in old_tracks:
            old_tracklet.show_history(frame)
            
        for new_tracklet in new_tracks:
            new_tracklet.show_history(frame)

        # show the boundaries
        frame_bbox.show(frame, (0,0,0))
        writer.write(frame)

    pbar.close()
    reader.release()
    writer.release()