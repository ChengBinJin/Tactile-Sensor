# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import time
import xlsxwriter
import numpy as np
import tensorflow as tf
from rg_dataset import Dataset
from resnet import ResNet18_Revised
from rg_solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('img_format', '.jpg', 'image format [.jpg | .png], default: .jpg')
tf.flags.DEFINE_string('domain', 'xy', 'data domain for [xy | rarb], default: xy')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize dataset
    data = Dataset(data='01' if FLAGS.domain == 'xy' else '02',
                   img_format=FLAGS.img_format,
                   is_train=False)

    # Initialize model
    model = ResNet18_Revised(input_shape=data.input_shape,
                             min_values=data.min_values,
                             max_values=data.max_values,
                             domain=FLAGS.domain,
                             is_train=False)

    # Initialize solver
    solver = Solver(model, data)

    # Initilize saver
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    # Decide the model folder
    if FLAGS.domain == 'xy':
        model_dir = '../model/20191111-094035'
    elif FLAGS.domain == 'rarb':
        model_dir = '../model/20191110-211525'
    else:
        raise NotImplementedError

    # Load trained model
    flag, iter_time = load_model(saver=saver, solver=solver, model_dir=model_dir)
    if flag is True:
        print(' [!] Load Success! Iter: {}'.format(iter_time))
    else:
        exit(' [!] Failed to restore model {}'.format(model_dir))

    tic = time.time()
    preds, gts = solver.test_eval(batch_size=1)
    total_pt = time.time() - tic
    avg_pt = total_pt / solver.data.num_test * 1000
    print(' [*] Avg. processing time: {:.3f} msec. {:.2f} FPS'.format(avg_pt, (1000. / avg_pt)))

    print(' [*] Writing excel...')
    write_to_csv(preds, gts, solver, model_dir=model_dir.split('/')[-1])
    print(' [!] Finished to write!')


def write_to_csv(preds, gts, solver, model_dir, save_folder='../result'):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Create a workbook and add a worksheet
    xlsx_name = os.path.join(save_folder, FLAGS.domain + '_data' + solver.data.data + '_' + model_dir + '.xlsx')
    workbook = xlsxwriter.Workbook(os.path.join('./', xlsx_name))
    xlsFormat = workbook.add_format()
    xlsFormat.set_align('center')
    xlsFormat.set_valign('vcenter')

    # Calculate l2 error and average error
    l2_error = np.sqrt(np.square(preds - gts))
    avg_error = np.mean(l2_error, axis=0)

    data_list = [('preds', preds), ('gts', gts), ('l2_error', l2_error)]
    attributes = ['No', 'Name', 'X', 'Y', 'Ra', 'Rb', 'F', 'D']
    for file_name, data in data_list:
        worksheet = workbook.add_worksheet(name=file_name)
        for attr_idx in range(len(attributes)):
            worksheet.write(0, attr_idx, attributes[attr_idx], xlsFormat)

        for idx in range(solver.data.num_test):
            for attr_idx in range(len(attributes)):
                if attr_idx == 0:       # No
                    worksheet.write(idx + 1, attr_idx, str(idx).zfill(3), xlsFormat)
                elif attr_idx == 1:     # Name
                    worksheet.write(idx + 1, attr_idx, solver.data.test_left_img_paths[idx], xlsFormat)
                else:
                    worksheet.write(idx + 1, attr_idx, data[idx, attr_idx - 2], xlsFormat)

        # Write average error
        if file_name == 'l2_error':
            worksheet.write(solver.data.num_test + 1, 1, 'average error', xlsFormat)
            for attr_idx in range(solver.data.num_attribute):
                worksheet.write(solver.data.num_test + 1, attr_idx + 2, avg_error[attr_idx], xlsFormat)

    workbook.close()


def load_model(saver, solver, model_dir, logger=None, is_train=False):
    if is_train:
        logger.info(' [*] Reading checkpoint...')
    else:
        print(' [*] Reading checkpoint...')

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(solver.sess, os.path.join(model_dir, ckpt_name))

        meta_graph_path = ckpt.model_checkpoint_path + '.meta'
        iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

        return True, iter_time
    else:
        return False, None


if __name__ == '__main__':
    tf.compat.v1.app.run()