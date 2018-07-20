import cv2
import argparse

from sgbm import SGBM
import utils as utils
from utils import RecordVideo, Reader

parser = argparse.ArgumentParser(description='')
parser.add_argument('--video_record', dest='video_record', action='store_true', default=False,
                    help='record raw video or not')
parser.add_argument('--input_video', dest='input_video', type=str, default='./videos/20180716-1733.avi',
                    help='input video')
args = parser.parse_args()


def main():
    stereo = SGBM([480, 640])
    reader = Reader(args.input_video)
    video_writer = RecordVideo(args.video_record)

    while True:
        # read fraems
        left_img, right_img = reader.next_frame()

        # show restuls
        utils.show_stereo(left_img, right_img, args.video_record, video_writer)
        utils.show_disparity(stereo, left_img, right_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everyting done, release the capture
    reader.turn_off()
    video_writer.turn_off()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
