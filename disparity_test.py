import numpy as np
import cv2


def main():
    left_cap = cv2.VideoCapture(0)   # 0: left camera
    right_cap = cv2.VideoCapture(1)  # 1: right camera

    while True:
        # Capture frame-by-frame
        left_ret, left_img = left_cap.read()
        right_ret, right_img = right_cap.read()

        if not left_ret and not right_ret:
            print('can not read frame from one of the camera')

        show_stereo(left_img, right_img)
        show_disparity(left_img, right_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everyting done, release the capture
    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()


def show_stereo(left_img, right_img):
    cv2.namedWindow('Stereo')
    cv2.moveWindow('Stereo', 0, 0)

    # Combine two imgs, height = 480, width = 640
    h, w, ch = left_img.shape
    img = np.zeros((h, 2 * w, ch), dtype=np.uint8)
    img[:, :w, :], img[:, w:2 * w, :] = left_img, right_img

    # Display the input frame
    cv2.imshow('Stereo', img)


def show_disparity(left_img, right_img):
    margin = 80
    cv2.namedWindow('Gray')
    cv2.moveWindow('Gray', 0, left_img.shape[0] + margin)

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # stereo = cv2.StereoBM_create(numDisparities=0, blockSize=15)
    min_disparity = 0
    num_disparity = 96  # maximum disparity minus minimum disparity
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparity, blockSize=5,
                                   preFilterCap=4, uniquenessRatio=5, speckleWindowSize=150, speckleRange=2,
                                   disp12MaxDiff=10, P1=600, P2=2400)
    # disparity = stereo.compute(gray_left, gray_right)
    disparity = stereo.compute(left_img, right_img)
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
