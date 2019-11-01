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
            self.model.gt_tfph: label_trains,
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
        print(' [*] Evalute on the validation dataset...')

        preds_total = np.zeros((self.data.num_val, self.data.num_attribute), dtype=np.float32)
        gts_total = np.zeros((self.data.num_val, self.data.num_attribute), dtype=np.float32)

        for i, index in enumerate(range(0, self.data.num_val, batch_size)):
            print('[{}/{}] processing...'.format(i + 1, (self.data.num_val // batch_size) + 1))

            img_vals, label_vals = self.data.direct_batch(batch_size=batch_size, start_index=index, stage='val')
            num_imgs = img_vals.shape[0]

            feed = {self.model.img_tfph: img_vals,
                    self.model.gt_tfph: label_vals}
            unnorm_preds, unnorm_gts = self.sess.run([self.model.unnorm_preds, self.model.unnorm_gts], feed_dict=feed)

            # Save unnormalized labels for using evaluation
            preds_total[i * batch_size :i * batch_size + num_imgs] = unnorm_preds
            gts_total[i * batch_size :i * batch_size + num_imgs] = unnorm_gts

        avg_err, summary = self.sess.run([self.model.avg_err, self.model.eval_summary_op],
                                         feed_dict={self.model.pred_tfph: preds_total,
                                                    self.model.gt_tfph: gts_total})

        return avg_err, summary
