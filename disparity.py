import cv2
import argparse

import utils as utils
from sgbm import SGBM
from detector import Detector
from utils import RecordVideo, Reader

parser = argparse.ArgumentParser(description='')
parser.add_argument('--video_record', dest='video_record', action='store_true', default=False,
                    help='record raw video or not')
parser.add_argument('--result_record', dest='result_record', action='store_true', default=False,
                    help='record result video or not')
parser.add_argument('--input_video', dest='input_video', type=str, default='./videos/20180716-1733.avi',
                    help='input video')
parser.add_argument('--interval_time', dest='interval_time', type=int, default=1,
                    help='interval time between two frames')
args = parser.parse_args()


def main():
    stereo = SGBM([480, 640])
    reader = Reader(args.input_video)
    video_writer = RecordVideo(args.video_record)
    detector = Detector()

    frame_idx = 0
    while True:
        # read fraems
        left_img, right_img = reader.next_frame()
        det_results = detector(left_img, right_img)
        imgs = [left_img, right_img, det_results['left_blob'], det_results['right_blob']]

        print('frame idx: {}'.format(frame_idx))

        # show restuls
        utils.show_stereo(imgs, args.video_record, video_writer)
        utils.show_disparity(stereo, det_results['mask'], det_results['left_thres'], det_results['right_thres'])

        if cv2.waitKey(args.interval_time) & 0xFF == 27:
            break

        frame_idx += 1

    # When everyting done, release the capture
    reader.turn_off()
    video_writer.turn_off()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
