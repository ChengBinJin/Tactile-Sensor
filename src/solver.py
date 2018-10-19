# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

import plot as plot
from dataset import DataLoader
from vgg16 import VGG16_TL


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = DataLoader(self.flags, path=self.flags.dataset)
        # self.model = VGG16_TL(self.sess, self.flags, self.dataset)

        self._make_folders()
        self.iter_time = 0

        # self.saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "{}_{}/model/{}".format(self.flags.dataset, self.flags.mode, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}_{}/model/{}".format(self.flags.dataset, self.flags.mode, cur_time)

            self.log_out_dir = "{}_{}/logs/{}".format(self.flags.dataset, self.flags.mode, cur_time)
            self.train_writer = tf.summary.FileWriter(self.log_out_dir, graph_def=self.sess.graph_def)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}_{}/model/{}".format(self.flags.dataset, self.flags.mode, self.flags.load_model)
            self.test_out_dir = "{}_{}/test/{}".format(self.flags.dataset, self.flags.mode, self.flags.load_model)
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
            # self.eval(self.iter_time)  # evaluate the

            imgs, gts = self.dataset.next_batch()
            print('imgs shape: {}'.format(imgs.shape))

            for idx in range(imgs.shape[0]):
                if self.flags.mode == 0:
                    left_img = (imgs[idx] + 1.) / 2.
                    cv2.imshow('show', left_img)
                    cv2.waitKey(0)
                elif self.flags.mode == 1:
                    img = imgs[idx]
                    left_img = img[:, :, 0]
                    right_img = img[:, :, 1]
                    show_img = np.zeros((left_img.shape[0], 2*left_img.shape[1]))
                    show_img[:, :left_img.shape[1]] = (left_img + 1.) / 2.
                    show_img[:, left_img.shape[1]:] = (right_img + 1.) / 2.
                    cv2.imshow('show', show_img)
                    cv2.waitKey(0)

            # loss, summary = self.model.train_step(imgs, gts)
            # self.model.print_info(loss, self.iter_time)
            # self.train_writer.add_summary(summary, self.iter_time)
            # self.train_writer.flush()
            #
            # # save model
            # self.save_model(self.iter_time)
            self.iter_time += 1

        # self.dataset.test_read_img()

    def test(self):
        print('Hello test!')

    def eval(self, iter_time):
        if np.mod(iter_time, self.flags.eval_freq) == 0:
            imgs, gts = self.dataset.next_batch_val()                   # sample val data
            preds = self.model.test_step(imgs)                          # predict
            unnorm_preds = self.dataset.un_normalize(preds)           # un-normalize predicts

            error = np.mean(np.sqrt(np.square(unnorm_preds - gts)), axis=0)   # calucate error rate
            print('error: {}'.format(error))
            print('gt: {}'.format(gts[0]))
            print('unnorm_preds: {}'.format(unnorm_preds[0]))
            print('preds: {}\n'.format(preds[0]))

            plot.plot('error', error)  # plot error
            plot.flush(self.log_out_dir)
            plot.tick()

    def save_model(self, iter_time):
        if np.mod(iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)
            print('[*] Model saved!')

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
