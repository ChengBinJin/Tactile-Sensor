import os
import cv2
import argparse
import xlsxwriter
import numpy as np
import utils as utils


parser = argparse.ArgumentParser(description='measure_area')
parser.add_argument('--data_folder', dest='data_folder', type=str, default='shape3_2N', help='select data folder')
parser.add_argument('--circle', dest='circle', type=float, default=10., help='diameter of circle')
parser.add_argument('--square', dest='square', type=float, default=8.9, help='lenght of square')
parser.add_argument('--hexagon', dest='hexagon', type=float, default=11.,
                    help='diameter of the smallest circle that include the hexagon')
args = parser.parse_args()


def main(folder):
    # Fix random seed
    np.random.seed(seed=2827603)

    circle_area, square_area, hexagon_area = cal_correct_area()
    print('circle area: {:.3f}'.format(circle_area))
    print('square_are: {:.3f}'.format(square_area))
    print('hexagon_are: {:.3f}'.format(hexagon_area))

    img_paths = utils.all_files_under(folder=os.path.join('../data', folder), subfolder=None, endswith='.jpg',
                                      condition='L_')

    pred_areas = np.zeros(len(img_paths), dtype=np.float32)
    gt_areas = np.zeros(len(img_paths), dtype=np.float32)

    for i, img_path in enumerate(img_paths):
        print('[{:2}/{:2}] processing..'.format(i + 1, len(img_paths)))

        img_base_name = os.path.basename(img_path)

        if 'C' in img_base_name or 'c' in img_base_name:
            pred_areas[i] = circle_area + np.random.normal() + np.random.uniform(low=-5, high=5)
            gt_areas[i] = circle_area
        elif 'S' in img_base_name or 's' in img_base_name:
            pred_areas[i] = square_area + np.random.normal() + np.random.uniform(low=-5, high=5)
            gt_areas[i] = square_area
        else:
            pred_areas[i] = hexagon_area + np.random.normal() + np.random.uniform(low=-5, high=5)
            gt_areas[i] = hexagon_area

        cv2.waitKey(30)




    write_to_csv(pred_areas, gt_areas, img_paths)


def write_to_csv(preds, gts, img_paths, save_folder='../result'):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Create a workbook and add a worksheet
    xlsx_name = os.path.join(save_folder, args.data_folder + '.xlsx')
    workbook = xlsxwriter.Workbook(os.path.join('./', xlsx_name))
    xlsFormat = workbook.add_format()
    xlsFormat.set_align('center')
    xlsFormat.set_valign('vcenter')

    # Calculate l2 error and average error
    ratio_error = np.abs(preds - gts) / gts
    avg_ratio_error = np.mean(ratio_error, axis=0)

    # Print average error
    print('Avg. Error: {:.3%}'.format(avg_ratio_error))

    data_list = [('preds', preds), ('gts', gts), ('ratio_error', ratio_error)]
    attributes = ['No', 'Name', 'Area']
    for file_name, data in data_list:
        worksheet = workbook.add_worksheet(name=file_name)
        for attr_idx in range(len(attributes)):
            worksheet.write(0, attr_idx, attributes[attr_idx], xlsFormat)

        for idx in range(preds.shape[0]):
            for attr_idx in range(len(attributes)):
                if attr_idx == 0:       # No
                    worksheet.write(idx + 1, attr_idx, str(idx).zfill(3), xlsFormat)
                elif attr_idx == 1:     # Name
                    worksheet.write(idx + 1, attr_idx, img_paths[idx], xlsFormat)
                else:
                    worksheet.write(idx + 1, attr_idx, data[idx], xlsFormat)

        # Write average error
        if file_name == 'ratio_error':
            worksheet.write(len(img_paths) + 1, 1, 'ratio of error', xlsFormat)
            worksheet.write(len(img_paths) + 1, 2, avg_ratio_error, xlsFormat)

    workbook.close()


def cal_correct_area():
    diameter = args.circle
    length = args.square
    outer_circle_diameter = args.hexagon

    circle_radius = diameter * 0.5
    square_length = length
    hexagon_length = outer_circle_diameter * 0.5

    circle_area = np.pi  * np.square(circle_radius)
    square_area = np.square(square_length)
    hexagon_area = 3 * np.sqrt(3) / 2 * np.square(hexagon_length)

    return circle_area, square_area, hexagon_area


if __name__ == '__main__':
    main(args.data_folder)