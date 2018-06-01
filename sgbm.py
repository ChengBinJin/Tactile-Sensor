import cv2
import numpy as np


class SGBM(object):
    def __init__(self, shape):
        self.min_disparity = 0
        num_disparities = (int((shape[1] / 8) + 15) & -16) - self.min_disparity
        num_channel = 1
        block_size = 7
        window_size = 3

        self.stereo = cv2.StereoSGBM_create(minDisparity=self.min_disparity, numDisparities=num_disparities,
                                            blockSize=block_size)
        self.stereo.setPreFilterCap(63)
        self.stereo.setP1(8 * num_channel * window_size**2)
        self.stereo.setP2(32 * num_channel * window_size**2)
        self.stereo.setUniquenessRatio(10)
        self.stereo.setSpeckleWindowSize(100)
        self.stereo.setSpeckleRange(2)
        self.stereo.setDisp12MaxDiff(1)
        # minDisparity: Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms
        #               can shift images, so this parameter needs to be adjusted accordingly.
        # numDisparities: Maximum disparity minus minimum disparity. The value is always greater than zero.
        #                 In the current implementation, this parameter must be divisible by 16.
        # blockSize: Matched block size. It must be an odd number >= 1. Normally, it should be somewhere in the
        #            3..11 range.
        # P1: The first parameter controlling the disparity smoothness. This parameter is used for the case of slanted
        #     surfaces (not fronto parallel).
        # P2: The second parameter controlling the dispairty smoothness. This parameter is used for "solving" the depth
        #     discontinuities problem. The larger the values are, the smoother the disparity is. P1 is the penalty on
        #     the disparity change by plus or minus 1 requires P2>P1. See stereo_match.cpp sample where some reasonable
        #     good P1 and P2 values are shown (like 8*number_of_image_channels*SAMWindowSize*SADWindowSize and
        #     32*number_of_image_channels*SADWindowSize*SADWindowSize, respectively).
        # disp12MaxDiff: Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it
        #                to a non-positive value to disable the check.
        # preFilterCap: Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at
        #               each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are
        #               passed to the Birchfield-Tomasi pixel cost function.
        # uniquenessRatio: Margin in percentage by which the best (minimum) computed cost function value should "win"
        #                  the second best value to consider the found match correct. Normally, a value within the 5-15
        #                  range is good enough.
        # speckleWindowSize: Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        #                    Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        # speckleRange: Maximum disparity variation within each connected component. If you do speckle filtering,
        #               set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or
        #               2 is good enough.
        # mode: Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will
        #       consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures.
        #       By default, it is set to false.

        self.disparity, self.norm_disparity, self.real_disparity = None, None, None

    def compute(self, left_img, right_img):
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        self.disparity = self.stereo.compute(gray_left, gray_right)
        # convert 12bit disparity to 8 bit integer and 4 bit float
        self.real_disparity = self.disparity.astype(np.float32) / 16.0

        return self.real_disparity

    def get_norm_disparity(self):
        # convert to integer and normalization
        # self.norm_disparity = (255. / self.stereo.getNumDisparities() * 16. * self.disparity).astype(np.uint8)
        self.norm_disparity = (self.real_disparity - self.min_disparity) / self.stereo.getNumDisparities()
        return self.norm_disparity


