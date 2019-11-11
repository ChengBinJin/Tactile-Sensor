# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
# import cv2
import numpy as np
import tensorflow as tf


class Solver(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, batch_size=4):
        img_trains, label_trains = self.data.train_random_batch(batch_size=batch_size)

        feed = {
            self.model.img_tfph: img_trains,
            self.model.label_tfph: label_trains,
            self.model.train_mode: True
        }

        train_op = self.model.train_op
        total_loss_op = self.model.total_loss
        data_loss_op = self.model.data_loss
        reg_term_op = self.model.reg_term
        summary_op = self.model.summary_op

        _, total_loss, data_loss, reg_term, summary = self.sess.run(
            [train_op, total_loss_op, data_loss_op, reg_term_op, summary_op], feed_dict=feed)

        return total_loss, data_loss, reg_term, summary

    def eval(self, batch_size=4):
        print(' [*] Evaluate on the validation dataset...')

        # Initialize/reset the running variables.
        self.sess.run(self.model.running_vars_initializer)

        for i, index in enumerate(range(0, self.data.num_val, batch_size)):
            img_vals, label_vals = self.data.direct_batch(batch_size=batch_size, start_index=index, stage='val')

            feed = {
                self.model.img_tfph: img_vals,
                self.model.label_tfph: label_vals,
                self.model.train_mode: False
            }

            _, pred_cls = self.sess.run([self.model.accuracy_metric_update, self.model.preds_cls], feed_dict=feed)
            print('Preds: {}'.format(pred_cls))
            print('label_val: {}'.format(label_vals))

        # Calculate the accuracy
        acc, summary = self.sess.run([self.model.accuracy_metric, self.model.metric_summary_op])

        return acc, summary
