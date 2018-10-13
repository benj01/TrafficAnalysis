import numpy as np
from uuid import uuid4
from tracking import random_color

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Finished = 4


class Track:
    def __init__(self, detection, max_age, n_init,sigma_h, color=None):
        self.color = color if color is not None else random_color()
        self.detections = [detection]
        self.ious = []
        self.id = uuid4()
        self._max_age = max_age
        self._n_init = n_init
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.sigma_h = sigma_h


    def predict(self):
        # does nothing because this is a simply IOU tracker with no propagation
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        self.ious.append(detection.iou(self.detections[-1]))
        self.detections.append(detection)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif max(self.ious) > self.sigma_h:
            self.state = TrackState.Finished
        else:
            self.state = TrackState.Deleted

    def get_ious(self, detections):
        return [self.detections[-1].iou(x) for x in detections]

    def show_history(self, frame, width=2, n=30):
        if self.is_confirmed():
            self.detections[-1].show(frame, self.color, width=width)

            for b in self.detections[-n:]:
                b.show_center(frame, self.color, width=-1)

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted


    def is_finished(self):
        return self.state == TrackState.Finished


class IOUTracker:
    def __init__(self, sigma_iou_discard=0.05, sigma_iou=0.4, sigma_h=0.7, max_age=2, n_init=3):
        self.sigma_iou_discard = sigma_iou_discard
        self.sigma_iou = sigma_iou # minimal to be consider as overlapping
        self.sigma_h = sigma_h
        self.max_age = max_age
        self.n_init = n_init
        self.finished_tracks = []
        self.active_tracks = []

    def predict(self):
        for track in self.active_tracks:
            track.predict()

    def update(self, detections, frame_bbox):
        for track in self.active_tracks:
            ious = np.array(track.get_ious(detections))

            if len(ious) == 0:
                track.mark_missed()
            else:
                i = np.argmax(ious)
                if ious[i] >= self.sigma_iou:
                    track.update(detections[i])
                    detections.remove(detections[i])
                else:
                    track.mark_missed()

        for detection in detections:
            ious = detection.get_ious([track.detections[-1] for track in self.active_tracks])
            # skip those that sufficiently overlap with existing active tracks
            if not np.any(np.array(ious) > self.sigma_iou_discard) and detection.is_inside(frame_bbox):
                self.active_tracks.append(Track(detection, self.max_age, self.n_init, self.sigma_h))

        tracks_finished = [track for track in self.active_tracks if track.is_finished()]
        tracks_deleted = [track for track in self.active_tracks if track.is_deleted()]

        self.finished_tracks += tracks_finished

        for track in tracks_finished + tracks_deleted:
            self.active_tracks.remove(track)
