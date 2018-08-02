import cv2
import numpy as np

from utils import RecordVideo


class Detector(object):
    def __init__(self, args):
        self.args = args
        self.mask_writer = RecordVideo(self.args.result_record, vname='mask')

        self.red = (0, 0, 255)
        self.white = (255, 255, 255)
        self.radius = 5
        self.threshold = 127.5
        self.kernel = np.ones((5, 5), np.uint8)

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
        left_blob = cv2.drawKeypoints(left_thres, left_keypoints, np.array([]), self.red,
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        right_blob = cv2.drawKeypoints(right_thres, right_keypoints, np.array([]), self.red,
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
