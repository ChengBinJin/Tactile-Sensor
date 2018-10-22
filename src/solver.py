# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import math
import xlsxwriter
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

# import plot as plot
from dataset import DataLoader
from vgg16 import VGG16_TL


logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_time, self.eval_time = 0, 0
        self.best_avg_err = math.inf
        self._make_folders()
        self._init_logger()

        self.dataset = DataLoader(self.flags, dataset_path=self.flags.dataset, log_path=self.log_out_dir)
        self.model = VGG16_TL(self.sess, self.flags, self.dataset, log_path=self.log_out_dir)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "results/{}_mode{}/model/{}".format(self.flags.dataset, self.flags.mode, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "results/{}_mode{}/model/{}".format(self.flags.dataset, self.flags.mode, cur_time)

            self.log_out_dir = "results/{}_mode{}/logs/{}".format(self.flags.dataset, self.flags.mode, cur_time)
            self.train_writer = tf.summary.FileWriter(self.log_out_dir, graph_def=self.sess.graph_def)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "results/{}_mode{}/model/{}".format(
                self.flags.dataset, self.flags.mode, self.flags.load_model)
            self.test_out_dir = "results/{}_mode{}/test/{}".format(
                self.flags.dataset, self.flags.mode, self.flags.load_model)

            cur_time = self.flags.load_model
            self.log_out_dir = "results/{}_mode{}/logs/{}".format(self.flags.dataset, self.flags.mode, cur_time)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def _init_logger(self):
        if self.flags.is_train:
            formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
            # file handler
            file_handler = logging.FileHandler(os.path.join(self.log_out_dir, 'solver.log'))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            # stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            # add handlers
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

            logger.info('gpu_index: {}'.format(self.flags.gpu_index))
            logger.info('batch_size: {}'.format(self.flags.batch_size))
            logger.info('resize_ratio: {}'.format(self.flags.resize_ratio))
            logger.info('num_regress: {}'.format(self.flags.num_regress))

            logger.info('mode: {}'.format(self.flags.mode))
            logger.info('dataset: {}'.format(self.flags.dataset))
            logger.info('is_train: {}'.format(self.flags.is_train))
            logger.info('learning_rate: {}'.format(self.flags.learning_rate))
            logger.info('weight_decay: {}'.format(self.flags.weight_decay))

            logger.info('iters: {}'.format(self.flags.iters))
            logger.info('print_freq: {}'.format(self.flags.print_freq))
            logger.info('eval_freq: {}'.format(self.flags.eval_freq))
            logger.info('load_model: {}'.format(self.flags.load_model))

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                logger.info('[*] Load SUCCESS!\n')
            else:
                logger.info('[! Load Failed...\n')

        while self.iter_time < self.flags.iters:
            imgs, gts = self.dataset.next_batch()
            loss, summary = self.model.train_step(imgs, gts)
            self.model.print_info(loss, self.iter_time)
            self.train_writer.add_summary(summary, self.iter_time)
            self.train_writer.flush()

            avg_error = self.eval(self.iter_time)  # evaluate the
            if avg_error < self.best_avg_err:
                self.best_avg_err = avg_error
                self.save_model(self.iter_time)  # save model

            self.iter_time += 1

        # self.dataset.test_read_img()

    def test(self):
        if self.load_model():
            logger.info(' [*] Load SUCCESS!')
        else:
            logger.info(' [!] Load Failed...')

        self.test_eval()  # evaluation

    def eval(self, iter_time):
        avg_error = math.inf
        if np.mod(iter_time, self.flags.eval_freq) == 0:
            print(' [*] Evaluate...')
            num_idxs = int(self.dataset.num_vals / self.flags.batch_size)
            preds_total = np.zeros((num_idxs * self.flags.batch_size, self.flags.num_regress), dtype=np.float32)
            gts_total = np.zeros((num_idxs * self.flags.batch_size, self.flags.num_regress), dtype=np.float32)

            for idx in range(num_idxs):
                imgs, gts = self.dataset.next_batch_val(idx)  # sample val data
                preds = self.model.test_step(imgs)  # predict
                preds_total[idx * self.flags.batch_size:idx * self.flags.batch_size + preds.shape[0], :] = preds
                gts_total[idx * self.flags.batch_size:idx * self.flags.batch_size + gts.shape[0], :] = gts

            unnorm_preds_total = self.dataset.un_normalize(preds_total)  # un-normalize predicts
            avg_error, summary = self.model.eval_step(unnorm_preds_total, gts_total)
            self.train_writer.add_summary(summary, self.eval_time)
            self.train_writer.flush()
            self.eval_time += 1

        return avg_error

    def test_eval(self):
        print(' [*] Evaluate...')
        preds_total = np.zeros((self.dataset.num_tests, self.flags.num_regress), dtype=np.float32)
        gts_total = np.zeros((self.dataset.num_tests, self.flags.num_regress), dtype=np.float32)
        num_idxs = int(np.ceil(self.dataset.num_tests / self.flags.batch_size))

        total_pt = 0.
        for idx in range(num_idxs):
            print('[{}] / [{}]'.format(idx+1, num_idxs))
            imgs, gts = self.dataset.next_batch_test(idx)  # sample test data

            tic = time.time()
            preds = self.model.test_step(imgs)  # predict
            total_pt += time.time() - tic

            preds_total[idx*self.flags.batch_size:idx*self.flags.batch_size+preds.shape[0], :] = preds
            gts_total[idx*self.flags.batch_size:idx*self.flags.batch_size+gts.shape[0], :] = gts

        unnorm_preds_total = self.dataset.un_normalize(preds_total)  # un-normalize predicts
        # errors = np.sqrt(np.square(unnorm_preds_total - gts_total))
        # print('errors shape: {}'.format(errors.shape))

        print(' [*] Avg. processing time: {:.3f} ms.'.format(total_pt / self.dataset.num_tests * 1000))

        self.write_to_csv(unnorm_preds_total, gts_total)  # write to xlsx file

    def write_to_csv(self, preds, gts):
        # Create a workbook and add a worksheet
        workbook = xlsxwriter.Workbook(os.path.join(self.test_out_dir, 'compare.xlsx'))
        xlsFormat = workbook.add_format()
        xlsFormat.set_align('center')
        xlsFormat.set_valign('vcenter')

        data_list = [('preds', preds), ('gts', gts), ('abs_error', np.abs(preds - gts))]
        attributes = ['No', 'Name', 'X', 'Y', 'Z', 'Ra', 'Rb', 'F', 'D']
        for file_name, data in data_list:
            worksheet = workbook.add_worksheet(name=file_name)
            for attr_idx in range(len(attributes)):
                worksheet.write(0, attr_idx, attributes[attr_idx], xlsFormat)

            for idx in range(self.dataset.num_tests):
                for attr_idx in range(len(attributes)):
                    if attr_idx == 0:
                        worksheet.write(idx+1, attr_idx, str(idx).zfill(3), xlsFormat)
                    elif attr_idx == 1:
                        worksheet.write(idx+1, attr_idx, self.dataset.test_left_paths[idx], xlsFormat)
                    else:
                        worksheet.write(idx+1, attr_idx, '{:.2f}'.format(data[attr_idx-2, 0]), xlsFormat)

    def save_model(self, iter_time):
        model_name = 'model'
        self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)
        logger.info(' [*] Model saved! Avg. error: {}!\n'.format(self.best_avg_err))

    def load_model(self):
        logger.info(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])
            logger.info(' [*] Load iter_time: {}'.format(self.iter_time))
            return True
        else:
            return False
