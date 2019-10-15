"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
# import os.path
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io
from scipy.optimize import linear_sum_assignment
# import glob
# import time
# import argparse
from filterpy.kalman import KalmanFilter


@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    static_count = 0  # Static variable to keep a count on number of internal IDs used so far

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        self.tracker_id = KalmanBoxTracker.static_count  # Tracker's ID
        KalmanBoxTracker.static_count += 1

        # define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement function
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0]])

        # measurement uncertainty/noise
        self.kf.R[2:, 2:] *= 10.
        # covariance matrix
        # P is already an eye matrix
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # Process uncertainty/noise
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01

        # filter state estimate
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        # Count of past consecutive frames for which detected box didn't match with this tracker

        self.count_successive_drops = 0  # Count of past successive frames where no update() was performed
        self.count_updates = 0  # Total number of update() performed on this tracker

    def update_with_box(self, bbox):
        """
        This function is called whenever a match is found with detected bounding box
        Updates the state vector with observed bbox.
        """
        self.count_updates += 1
        self.count_successive_drops = 0  # Reset since the succession is broken
        self.kf.update(convert_bbox_to_z(bbox))

    def update_without_box(self):
        """
        No bounding box was detected for this tracker. Mark this update.
        :return:
        """
        self.count_successive_drops += 1

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_rows, match_cols = linear_sum_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_rows:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in match_cols:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for matched_row, matched_col in zip(matched_rows, match_cols):
        if iou_matrix[matched_row, matched_col] < iou_threshold:
            unmatched_detections.append(matched_row)
            unmatched_trackers.append(matched_col)
        else:
            matches.append(((matched_row, matched_col),))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        # Limit which every tracker must cross before getting acknowledged as true new object
        # Should be low enough to quickly acknowledge new objects entering into the frame
        # Should be high enough to delay acknowledgement of new tracklets getting created due to larger displacements
        self.min_hits = min_hits

        # If a box is not detected by this limit, then we will forget it (terminate its tracklet)
        # Should be long enough to accommodate minor glitches
        self.max_age = max_age  # Count of absent frames after which to forget the passenger

        # In case, Sort() mistakenly creates a new tracklet due to larger displacement and there's nothing
        # that we can do about it, then the old tracker must be terminated before acknowledging
        # the new tracker as valid passenger.
        # This is needed to make certain that overall count of tracked objects stays correct despite this mistake
        assert self.max_age < self.min_hits

        self._object_counter = 0
        self.trackers = list()

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # get predicted locations from existing trackers.
        tracked_boxes = np.zeros((len(self.trackers), 5))
        to_del = []  # List of trackers to be removed
        ret = []  # List of bounding boxes to be returned

        # Loop over all tracked bounding boxes
        for t, tracked_box in enumerate(tracked_boxes):
            pos = self.trackers[t]['tracker'].predict()[0]  # Run tracker prediction
            tracked_box[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):  # TODO: why is this needed?
                to_del.append(t)

        tracked_boxes = np.ma.compress_rows(np.ma.masked_invalid(tracked_boxes))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, tracked_boxes)

        # Update existing trackers
        for t, trk_dict in enumerate(self.trackers):
            if t in unmatched_trks:
                trk_dict['tracker'].update_without_box()
            else:
                # update matched trackers with assigned detections
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk_dict['tracker'].update_with_box(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append({'object_id': None, 'tracker': trk})

        to_del = list()  # List of tracklets to be terminated
        for i in range(len(self.trackers)):
            if self.trackers[i]['tracker'].count_successive_drops >= self.max_age:
                to_del.append(i)
            elif self.trackers[i]['tracker'].count_updates >= self.min_hits:
                d = self.trackers[i]['tracker'].get_state()[0]
                if self.trackers[i]['object_id'] is None:
                    self.trackers[i]['object_id'] = self._object_counter
                    self._object_counter += 1

                ret.append(np.concatenate((d, [self.trackers[i]['object_id']])).reshape(1, -1))

        # Terminate dead tracklet
        for i in reversed(to_del):
            self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 5))





