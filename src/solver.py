# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import math
import numpy as np
import tensorflow as tf
from datetime import datetime

# import plot as plot
from dataset import DataLoader
from vgg16 import VGG16_TL


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_time, self.eval_time = 0, 0
        self.best_avg_err = math.inf
        self._make_folders()

        self.dataset = DataLoader(self.flags, dataset_path=self.flags.dataset, log_path=self.log_out_dir)
        self.model = VGG16_TL(self.sess, self.flags, self.dataset)

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

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                print('[*] Load SUCCESS!\n')
            else:
                print('[! Load Failed...\n')

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
        print('Hello test!')

    def eval(self, iter_time):
        avg_error = math.inf
        if np.mod(iter_time, self.flags.eval_freq) == 0:
            preds_total, gts_total = [], []
            num_idxs = int(self.dataset.num_vals / self.flags.batch_size)
            for idx in range(num_idxs):
                print('{} / {}'.format(idx+1, num_idxs))
                imgs, gts = self.dataset.next_batch_val(idx)  # sample val data
                preds = self.model.test_step(imgs)  # predict
                preds_total.append(preds)
                gts_total.append(gts)

            preds_total = np.asarray(preds_total).reshape((-1, 7))
            gts_total = np.asarray(gts_total).reshape((-1, 7))

            # unnorm_preds = self.dataset.un_normalize(preds_total)  # un-normalize predicts
            # error = np.mean(np.sqrt(np.square(unnorm_preds - gts_total)), axis=0)  # calucate error rate
            # print('python error: {}'.format(error))

            avg_error, summary = self.model.eval_step(preds_total, gts_total)
            self.train_writer.add_summary(summary, self.eval_time)
            self.train_writer.flush()

            self.eval_time += 1

        return avg_error

    def save_model(self, iter_time):
        model_name = 'model'
        self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)
        print('[*] Model saved! Avg. error: {}'.format(self.best_avg_err))

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            print('[*] Load iter_time: {}'.format(self.iter_time))
            return True
        else:
            return False
