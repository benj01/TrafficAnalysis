from detectionprovider import DetectionProvider
from tracking import Detection
from IOUTracker import IOUTracker
from video import VideoStreamReader, VideoStreamWriter, FileListReader
from yolo import YOLO
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


if __name__ == "__main__":
    filename = "4K Traffic camera video - free download now!-MNn9qKG2UFI.webm"
    seconds_skip = 0  # number of seconds to skip
    seconds_count = 60  # number of seconds to process
    # reader = FileListReader("/home/xmichaelx/Pobrane/MVI_20011/", seconds_count=seconds_count, seconds_skip=seconds_skip, fps=30)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    set_session(tf.Session(config=config))

    yolo = YOLO()

    reader = VideoStreamReader(filename, seconds_count=seconds_count, seconds_skip=seconds_skip, width=1920, height=1080)
    writer = VideoStreamWriter("iou.avi", width=reader.width, height=reader.height, fps=reader.fps)
    empty_writer = VideoStreamWriter("iou_empty.avi", width=reader.width, height=reader.height, fps=reader.fps)
    detection_provider = DetectionProvider(yolo)
    frame_bbox = Detection.from_frame((reader.width,reader.height), 80)
    tracker = IOUTracker()

    pbar = tqdm(total=reader.frame_count - reader.frame_skip)

    while True:
        frame = reader.next_frame()

        if frame is None:
            break

        pbar.update()

        detections = detection_provider.detect_boxes(frame, reader.frame_no)
        # for detection in detections:
        #     detection.show(frame, (255, 255, 255))

        tracker.predict()
        tracker.update(detections, frame_bbox)

        zero_frame = np.zeros_like(frame)
        for track in tracker.active_tracks:
            track.show_history(zero_frame)
            track.show_history(frame)


        frame_bbox.show(frame, (0,0,0))
        writer.write(frame)
        empty_writer.write(zero_frame)

    pbar.close()
    reader.release()
    writer.release()
    empty_writer.release()
