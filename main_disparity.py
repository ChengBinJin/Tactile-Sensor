import cv2
import argparse
import numpy as np

import utils as utils
from sgbm import SGBM
from detector import Detector
from utils import RecordVideo, Reader

parser = argparse.ArgumentParser(description='')
parser.add_argument('--video_record', dest='video_record', action='store_true', default=False,
                    help='record raw video or not')
parser.add_argument('--result_record', dest='result_record', action='store_true', default=False,
                    help='record result video or not')
parser.add_argument('--tracker', dest='tracker', action='store_true', default=False,
                    help='use tracker or not')
parser.add_argument('--sparsity', dest='sparsity', type=int, default=1, help='tracker sparsity for acceleration')
parser.add_argument('--input_video', dest='input_video', type=str, default='./videos/20180716-1733.avi',
                    help='input video')
parser.add_argument('--interval_time', dest='interval_time', type=int, default=1,
                    help='interval time between two frames')
args = parser.parse_args()

H, W = 480, 640


def main():
    stereo = SGBM([H, W])
    reader = Reader(args.input_video)
    video_writer = RecordVideo(args.video_record, height=H, width=2*W)
    blob_writer = RecordVideo(args.result_record, height=2*H, width=2*W, vname='blob')
    disp_writer = RecordVideo(args.result_record, height=2*H, width=2*W, vname='disp')
    detector = Detector(args)

    frame_idx = 0
    while True:
        if np.mod(frame_idx, 2) == 0:
            # read fraems
            left_img, right_img = reader.next_frame()
            if (left_img is None) or (right_img is None):
                break

            # blob detection
            det_results = detector(left_img, right_img)
            imgs = [left_img, right_img, det_results['left_blob'], det_results['right_blob']]

            # show restuls
            utils.show_stereo(imgs, args, video_writer, blob_writer)
            utils.show_disparity(stereo, args, det_results, disp_writer)

            if cv2.waitKey(args.interval_time) & 0xFF == 27:
                break

        print('frame idx: {}'.format(frame_idx))
        frame_idx += 1

    # When everyting done, release the capture
    print('Turn off the recordings')
    reader.turn_off()
    video_writer.turn_off()
    blob_writer.turn_off()
    disp_writer.turn_off()
    detector.mask_writer.turn_off()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
