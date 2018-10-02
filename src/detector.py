import cv2
import numpy as np

from tracker import Tracker
from utils import RecordVideo


class Detector(object):
    def __init__(self, args):
        self.args = args
        self.mask_writer = RecordVideo(self.args.result_record, vname='mask')
        self.left_tracker = Tracker(max_age=20, min_hits=5)
        self.right_tracker = Tracker(max_age=20, min_hits=5)
        self.sparsity = self.args.sparsity  # accelerate tracking

        self.red = (0, 0, 255)
        self.white = (255, 255, 255)
        self.aqua = (255, 255, 0)
        self.myFont = 2
        self.myScale = 0.5
        self.radius_factor = 1.2
        self.threshold = 127.5
        self.kernel = np.ones((5, 5), np.uint8)
        self.width, self.height, self.channel = 640, 480, 3

        # Set up the SimpleBlobdetector with default parameters
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255

        # Change thresholds
        params.minThreshold = 200
        params.maxThreshold = 256

        # Filter by Area
        params.filterByArea = True
        params.minArea = 30

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        # params.minConvexity = 0.01

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1

        self.detector = cv2.SimpleBlobDetector_create(params)

    def __call__(self, left_img, right_img):
        # BGR to gray
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Thresholding
        ret1, left_thres = cv2.threshold(gray_left, self.threshold, 255, cv2.THRESH_BINARY)
        ret2, right_thres = cv2.threshold(gray_right, self.threshold, 255, cv2.THRESH_BINARY)

        # Dilation and Erosin
        left_thres = cv2.morphologyEx(left_thres, cv2.MORPH_OPEN, self.kernel)
        right_thres = cv2.morphologyEx(right_thres, cv2.MORPH_OPEN, self.kernel)
        det_results = {'left_thres': left_thres, 'right_thres': right_thres}

        # Detect blobs
        left_keypoints = self.detector.detect(left_thres)
        right_keypoints = self.detector.detect(right_thres)
        det_results['left_key'], det_results['right_key'] = left_keypoints, right_keypoints

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensure the size of the circle corresponds to the size of blob
        # left_blob = cv2.drawKeypoints(left_thres, left_keypoints, np.array([]), self.red,
        #                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # right_blob = cv2.drawKeypoints(right_thres, right_keypoints, np.array([]), self.red,
        #                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        left_blob = np.dstack((left_thres, left_thres, left_thres))
        right_blob = np.dstack((right_thres, right_thres, right_thres))

        if self.args.tracker:
            # kalman filter with hungarian data association
            left_dets, right_dets = self.list_to_array(left_keypoints, right_keypoints)
            left_tracks = self.left_tracker.update(left_dets)
            right_tracks = self.right_tracker.update(right_dets)
            # print('left_tracks 4th dim value: {}'.format(left_tracks[:, 4]))

            for idx in range(left_tracks.shape[0]):
                col = int((left_tracks[idx, 0] + left_tracks[idx, 2]) / 2.)
                row = int((left_tracks[idx, 1] + left_tracks[idx, 3]) / 2.)
                radius = int(self.radius_factor * (left_tracks[idx, 2] - left_tracks[idx, 0]) / 2.)
                id_ = str(int(left_tracks[idx, 4]))
                cv2.circle(left_blob, (col, row), radius, color=self.aqua, thickness=-1)
                left_blob[row - 1:row + 1, col - 1:col + 1, :] = [0, 0, 255]
                cv2.putText(left_blob, id_, (col, row), fontFace=self.myFont, fontScale=self.myScale, color=self.red)
            for idx in range(right_tracks.shape[0]):
                col = int((right_tracks[idx, 0] + right_tracks[idx, 2]) / 2.)
                row = int((right_tracks[idx, 1] + right_tracks[idx, 3]) / 2.)
                radius = int(self.radius_factor * (right_tracks[idx, 2] - right_tracks[idx, 0]) / 2.)
                id_ = str(int(right_tracks[idx, 4]))
                cv2.circle(right_blob, (col, row), radius, color=self.aqua, thickness=-1)
                right_blob[row - 1:row + 1, col - 1:col + 1, :] = [0, 0, 255]
                cv2.putText(right_blob, id_, (col, row), fontFace=self.myFont, fontScale=self.myScale, color=self.red)
        else:
            # add keypoints' center and draw circle
            for idx in range(len(left_keypoints)):
                col, row = int(left_keypoints[idx].pt[0]), int(left_keypoints[idx].pt[1])
                radius = int(self.radius_factor * left_keypoints[idx].size / 2)
                cv2.circle(left_blob, (col, row), radius, color=self.aqua, thickness=-1)
                left_blob[row - 1:row + 1, col - 1:col + 1, :] = [0, 0, 255]
            for idx in range(len(right_keypoints)):
                col, row = int(right_keypoints[idx].pt[0]), int(right_keypoints[idx].pt[1])
                radius = int(self.radius_factor * right_keypoints[idx].size / 2)
                cv2.circle(right_blob, (col, row), radius, color=self.aqua, thickness=-1)
                right_blob[row - 1:row + 1, col - 1:col + 1, :] = [0, 0, 255]

        det_results['left_blob'], det_results['right_blob'] = left_blob, right_blob
        left_mask = self.generate_mask(left_thres, left_keypoints)
        det_results['mask'] = left_mask
        cv2.imshow('Left Mask', left_mask)

        if self.args.result_record is True:
            # the frame for recording have to be dimension equal to 3
            self.mask_writer.output.write(np.dstack((left_mask, left_mask, left_mask)))

        return det_results

    def generate_mask(self, left_gray, left_keypoints):
        mask = np.zeros_like(left_gray)

        for keypoint in left_keypoints:
            cv2.circle(mask, center=(int(keypoint.pt[0]), int(keypoint.pt[1])),
                       radius=int(np.floor(keypoint.size / 2.)), color=self.white, thickness=-1)

        return mask

    def list_to_array(self, left_keypoints, right_keypoints):
        left_dets = np.zeros((int(np.ceil(len(left_keypoints) / self.sparsity)), 5), dtype=np.float32)
        right_dets = np.zeros((int(np.ceil(len(right_keypoints) / self.sparsity)), 5), dtype=np.float32)

        # save as (x1, y1, x2, y2)
        idx = 0
        for i in range(0, len(left_keypoints), self.sparsity):
            radius = left_keypoints[idx].size / 2.
            left_dets[idx, 0] = np.maximum(0, left_keypoints[i].pt[0] - radius)
            left_dets[idx, 1] = np.maximum(0, left_keypoints[i].pt[1] - radius)
            left_dets[idx, 2] = np.minimum(left_keypoints[i].pt[0] + radius, self.width)
            left_dets[idx, 3] = np.minimum(left_keypoints[i].pt[1] + radius, self.height)
            idx += 1

        idx = 0
        for i in range(0, len(right_keypoints), self.sparsity):
            radius = right_keypoints[idx].size / 2.
            right_dets[idx, 0] = np.maximum(0, right_keypoints[i].pt[0] - radius)
            right_dets[idx, 1] = np.maximum(0, right_keypoints[i].pt[1] - radius)
            right_dets[idx, 2] = np.minimum(right_keypoints[i].pt[0] + radius, self.width)
            right_dets[idx, 3] = np.minimum(right_keypoints[i].pt[1] + radius, self.height)
            idx += 1

        return left_dets, right_dets
