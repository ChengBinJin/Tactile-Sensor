# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from datetime import datetime


from dataset import DataLoader
from vgg16 import VGG16_TL


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = DataLoader(self.flags, path=self.flags.dataset)
        self.model = VGG16_TL(self.sess, self.flags, self.dataset.img_size)

        self._make_folders()
        self.iter_time = 0

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)

            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time),
                                                      graph_def=self.sess.graph_def)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
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
            # print('imgs shape: {}'.format(imgs.shape))
            # print('gts shape: {}'.format(gts.shape)

            loss, summary = self.model.train_step(imgs, gts)
            self.model.print_info(loss, self.iter_time)
            self.train_writer.add_summary(summary, self.iter_time)
            self.train_writer.flush()

            self.iter_time += 1

        # self.dataset.test_read_img()

    def test(self):
        print('Hello test!')

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
