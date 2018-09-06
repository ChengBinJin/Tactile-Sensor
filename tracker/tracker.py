from __future__ import print_function

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter


def iou_tracker(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])  # x1
    yy1 = np.maximum(bb_test[1], bb_gt[1])  # x2
    xx2 = np.minimum(bb_test[2], bb_gt[2])  # x3
    yy2 = np.minimum(bb_test[3], bb_gt[3])  # x4
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)  # iou
    return iou


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2
    y = bbox[1] + h/2
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))  # center_x, center_y, area, ratio


def convert_x_to_bbox(x, sccore=None):
    w = np.sqrt(x[2]*x[3])
    h = x[2] / w
    if sccore is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))  # x1, y1, x2, y2
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., sccore]).reshape((1, 5))


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou_tracker(trk, det)

    # Solve the linear assignment problem using the Hungarian algorithm
    # The problem is also known as maximum weight matching in bipartite graphs. The method is also known as the
    # Munkres or Kuhn-Munkres algorithm.
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Tracker(object):
    def __init__(self, max_age=5, min_hits=0, det_confidence=0.5):  
        self.max_age = max_age
        self.min_hits = min_hits
        self.det_confidence = det_confidence
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        if len(dets) != 0:  # (x, y, h, w) to (x1, y1, x2, y2)
            dets[:, 2] = dets[:, 2] + dets[:, 0]
            dets[:, 3] = dets[:, 3] + dets[:, 1]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret1 = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):  # np.isnan: test element-wise for NaN and return result as a bollean array
                to_del.append(t)

        # np.ma.masked_invalid: Mask an array where invalid values occur (NaNs or infs)
        # np.ma.compress_rows: Suppresss whole rows of a 2-D array that contain masked values
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Hungarian data association
        matched, unmateched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        for i in unmateched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < self.max_age) and \
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret1.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret1) > 0:
            return np.concatenate(ret1)
        else:
            return np.empty((0, 5))
