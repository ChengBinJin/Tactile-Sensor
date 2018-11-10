import os
import argparse
import xlrd
import xlsxwriter
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='train03', help='dataset for train03 or train04')
parser.add_argument('--mode', type=int, default=1, help='mode for 0, 1, 2, 3')
parser.add_argument('--model', type=str, default='20181024-2108', help='model name is time')
args = parser.parse_args()

num_regress = 7
save_path = None


def main():
    filename = '{}_mode{}'.format(args.dataset, str(args.mode))
    global save_path
    save_path = './results/{}/test/{}'.format(filename, args.model)
    csv_path = './results/{}/test/{}/{}_mode{}.xlsx'.format(filename, args.model, args.dataset, str(args.mode).zfill(2))
    print('csv path: {}'.format(csv_path))

    gts_arr, preds_arr, img_paths = read_xlsx(csv_path)

    # save csv
    eval03(gts_arr, preds_arr, img_paths)
    eval05(gts_arr, preds_arr, img_paths)
    # eval06(gts_arr, preds_arr, img_paths)
    # eval09(gts_arr, preds_arr, img_paths)
    # eval10(gts_arr, preds_arr, img_paths)
    # eval11(gts_arr, preds_arr, img_paths)


def read_xlsx(path):
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_name('preds')

    # read gts and preds values
    preds_arr = np.zeros((worksheet.nrows-2, num_regress), dtype=np.float64)
    gts_arr = np.zeros_like(preds_arr)
    img_paths = []
    for row_idx in range(1, worksheet.nrows-1):
        # init gt_arrays
        file_name = worksheet.cell(row_idx, 1).value
        img_paths.append(file_name)

        gts_arr[row_idx-1, 0] = float(file_name[file_name.find('_X') + 2:file_name.find('_Y')])
        gts_arr[row_idx-1, 1] = float(file_name[file_name.find('_Y') + 2:file_name.find('_Z')])
        gts_arr[row_idx-1, 2] = float(file_name[file_name.find('_Z') + 2:file_name.find('_Ra')])
        gts_arr[row_idx-1, 3] = float(file_name[file_name.find('_Ra') + 3:file_name.find('_Rb')])
        gts_arr[row_idx-1, 4] = float(file_name[file_name.find('_Rb') + 3:file_name.find('_F')])
        gts_arr[row_idx-1, 5] = float(file_name[file_name.find('_F') + 2:file_name.find('_D')])
        gts_arr[row_idx-1, 6] = float(file_name[file_name.find('_D') + 2:file_name.find('.bmp')])

        # if F is small than 0.1, all attributes are 0
        if gts_arr[row_idx, 5] < 0.1:
            gts_arr[row_idx, :] = 0
        # if D is smalle than zero, we set to 0
        if gts_arr[row_idx, 6] < 0:  # D
            gts_arr[row_idx, 6] = 0

        for col_idx in range(2, worksheet.ncols):
            preds_arr[row_idx-1, col_idx-2] = float(worksheet.cell(row_idx, col_idx).value)

    is_check_read = False
    if is_check_read:
        for idx in range(preds_arr.shape[0]):
            print('ID: {}'.format(idx))
            print('Img path: {}'.format(worksheet.cell(idx+1, 1)))
            print('gt: {}'.format(gts_arr[idx]))
            print('pred: {}\n'.format(preds_arr[idx]))

    return gts_arr, preds_arr, img_paths


def eval03(gts_arr, preds_arr, img_paths):
    filter_gts, filter_preds, filter_img_paths = [], [], []
    for idx, img_path in enumerate(img_paths):
        # select X or Y is 0
        if gts_arr[idx, 0] == 0 or gts_arr[idx, 1] == 0:
            filter_gts.append(gts_arr[idx, :2])
            filter_preds.append(preds_arr[idx, :2])
            filter_img_paths.append(img_path)

    filter_gts_arr = np.asarray(filter_gts)
    filter_preds_arr = np.asarray(filter_preds)

    is_check_read = False
    if is_check_read:
        for idx, img_path in enumerate(filter_img_paths):
            print('img_path: {}'.format(img_path))
            print('filter_gt: {}'.format(filter_gts_arr[idx]))
            print('filter_pred: {}'.format(filter_preds_arr[idx]))

    # calculate l2 error and average error
    l2_error = np.sqrt(np.square(filter_preds_arr - filter_gts_arr))
    avg_error = np.mean(l2_error, axis=0)

    data_list = [('preds', filter_preds_arr), ('gts', filter_gts_arr), ('l2_error', l2_error)]
    attributes = ['No', 'Name', 'X', 'Y']
    num_tests = filter_gts_arr.shape[0]
    file_name = os.path.join(save_path, 'eval03.xlsx')

    write_to_csv(filter_img_paths, data_list, attributes, avg_error, num_tests, file_name)


def eval05(gts_arr, preds_arr, img_paths):
    filter_gts, filter_preds, filter_img_paths = [], [], []
    for idx, img_path in enumerate(img_paths):
        # select X or Y is 0
        if (
                gts_arr[idx, 0] == -6. and gts_arr[idx, 1] == -6.) or (
                gts_arr[idx, 0] == -6. and gts_arr[idx, 1] == 6.) or (
                gts_arr[idx, 0] == 6. and gts_arr[idx, 1] == -6.) or (
                gts_arr[idx, 0] == 6. and gts_arr[idx, 1] == 6.
        ):
            filter_gts.append(gts_arr[idx, :2])
            filter_preds.append(preds_arr[idx, :2])
            filter_img_paths.append(img_path)

    filter_gts_arr = np.asarray(filter_gts)
    filter_preds_arr = np.asarray(filter_preds)

    is_check_read = True
    if is_check_read:
        for idx, img_path in enumerate(filter_img_paths):
            print('img_path: {}'.format(img_path))
            print('filter_gt: {}'.format(filter_gts_arr[idx]))
            print('filter_pred: {}'.format(filter_preds_arr[idx]))

    # calculate l2 error and average error
    l2_error = np.sqrt(np.square(filter_preds_arr - filter_gts_arr))
    avg_error = np.mean(l2_error, axis=0)

    data_list = [('preds', filter_preds_arr), ('gts', filter_gts_arr), ('l2_error', l2_error)]
    attributes = ['No', 'Name', 'X', 'Y']
    num_tests = filter_gts_arr.shape[0]
    file_name = os.path.join(save_path, 'eval05.xlsx')

    write_to_csv(filter_img_paths, data_list, attributes, avg_error, num_tests, file_name)


def eval06(gts_arr, preds_arr, img_paths):
    print(' [*] Hello eval06...')


def eval09(gts_arr, preds_arr, img_paths):
    print(' [*] Hello eval09...')


def eval10(gts_arr, preds_arr, img_paths):
    print(' [*] Hello eval10...')


def eval11(gts_arr, preds_arr, img_paths):
    print(' [*] Hello eval11...')


def write_to_csv(img_paths, data_list, attributes, avg_error, num_tests, file_name):
    # Create a workbook and add a worksheet
    workbook = xlsxwriter.Workbook(file_name)
    xlsFormat = workbook.add_format()
    xlsFormat.set_align('center')
    xlsFormat.set_valign('vcenter')

    for file_name, data in data_list:
        worksheet = workbook.add_worksheet(name=file_name)
        for attr_idx in range(len(attributes)):
            worksheet.write(0, attr_idx, attributes[attr_idx], xlsFormat)

        for idx in range(num_tests):
            for attr_idx in range(len(attributes)):
                if attr_idx == 0:
                    worksheet.write(idx + 1, attr_idx, str(idx).zfill(3), xlsFormat)
                elif attr_idx == 1:
                    worksheet.write(idx + 1, attr_idx, img_paths[idx], xlsFormat)
                else:
                    worksheet.write(idx + 1, attr_idx, data[idx, attr_idx - 2], xlsFormat)

        # write average error
        if file_name == 'l2_error':
            worksheet.write(num_tests + 1, 1, 'average error', xlsFormat)
            for attr_idx in range(len(attributes) - 2):
                worksheet.write(num_tests + 1, attr_idx + 2, avg_error[attr_idx], xlsFormat)


if __name__ == '__main__':
    main()
