import numpy as np
import cv2


def main():
    cv2.namedWindow('Left & Right')
    cv2.moveWindow('Left & Right', 0, 0)

    # read images
    left = cv2.imread('./img/cones_imL.png', 1)
    right = cv2.imread('./img/cones_imR.png', 1)

    h, w, ch = left.shape
    img = np.zeros((h, 2*w, ch), dtype=np.uint8)
    img[:, :w, :] = left.copy()
    img[:, w:2*w, :] = right.copy()

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    min_disparity = -16
    num_disparity = 80 - min_disparity
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparity, blockSize=5)
    disparity = stereo.compute(gray_left, gray_right)

    real_disparity = disparity.astype(np.float32) / 16.
    norm_disparity = (real_disparity - min_disparity) / num_disparity

    cv2.imshow('Left & Right', img)
    cv2.imshow('real_disparity', real_disparity)
    cv2.imshow('norm_disparity', norm_disparity)
    cv2.waitKey(0)


def show_disparity(left_img, right_img):
    margin = 80
    cv2.namedWindow('Gray')
    cv2.moveWindow('Gray', 0, left_img.shape[0] + margin)

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # stereo = cv2.StereoBM_create(numDisparities=0, blockSize=15)
    min_disparity = -64
    num_disparity = 32 - min_disparity
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparity, blockSize=5)
    disparity = stereo.compute(gray_left, gray_right)
    real_disparity = disparity.astype(np.float32) / 16.
    norm_disparity = (real_disparity - min_disparity) / num_disparity

    cv2.imshow('real_disparity', real_disparity)
    cv2.imshow('norm_disparity', norm_disparity)

    # Combine two imgs, height = 480, width = 640
    h, w = gray_left.shape
    img = np.zeros((h, 2 * w), dtype=np.uint8)
    img[:, :w], img[:, w:2 * w] = gray_left, gray_right

    # Display the input frame
    cv2.imshow('Gray', img)


if __name__ == '__main__':
    main()
