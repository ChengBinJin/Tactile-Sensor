import numpy as np
import cv2

left_cap = cv2.VideoCapture(0)   # 0: left camera
right_cap = cv2.VideoCapture(1)  # 1: right camera

while True:
    # Capture frame-by-frame
    left_ret, left_img = left_cap.read()
    right_ret, right_img = right_cap.read()

    if not left_ret and not right_ret:
        print('can not read frame from one of the camera')

    # Combine two imgs
    h, w, ch = left_img.shape
    print('height: {}, width: {}'.format(h, w))
    img = np.zeros((h, 2 * w, ch), dtype=np.uint8)
    img[:, :w, :], img[:, w:2*w, :] = left_img, right_img

    # Display the input frame
    cv2.imshow('Stereo', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everyting done, release the capture
left_cap.release()
right_cap.release()

cv2.destroyAllWindows()
