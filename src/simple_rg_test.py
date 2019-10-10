# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_utils as tf_utils


class ResNet18(object):
    def __init__(self, input_shape=(185, 242, 2), num_attribute=6, name='ResNet18'):
        self.input_shape = input_shape
        self.num_attribute = num_attribute
        self.name = name
        self.layers = [2, 2, 2, 2]
        # Initialize session
        self.sess = tf.compat.v1.Session()

        # Model should be fixed in here
        self.model_dir = '../model/20191009-213543'
        self.data = '01'

        self._build_graph()
        # self._init_variables()
        self._read_min_max_info()

        flag, iter_time = self.load_model()
        if flag is True:
            print(' [!] Load Success! Iter: {}'.format(iter_time))
        else:
            exit(' [!] Failed to restore model {}'.format(self.model_dir))

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build_graph(self):
        self.img_tfph = tf.compat.v1.placeholder(dtype=tf.dtypes.float32, shape=[None, *self.input_shape])

        # Network forward for training
        preds = self.forward_network(input_img=self.normalize(self.img_tfph), reuse=False)
        self.preds = tf.math.maximum(x=tf.zeros_like(preds, dtype=tf.dtypes.float32), y=preds)

    def predict(self, left_img, right_img):
        # Preprocessing
        pre_img = self.preprocessing(left_img, right_img)
        output = self.sess.run(self.preds, feed_dict={self.img_tfph: np.expand_dims(pre_img, axis=0)})
        return output

    def forward_network(self, input_img, reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            tf_utils.print_activations(input_img, logger=None)
            inputs = self.conv2d_fixed_padding(inputs=input_img, filters=64, kernel_size=7, strides=2, name='conv1')
            inputs = tf_utils.max_pool(inputs, name='3x3_maxpool', ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                       logger=None)

            inputs = self.block_layer(inputs=inputs, filters=64, block_fn=self.bottleneck_block, blocks=self.layers[0],
                                      strides=1, train_mode=False, name='block_layer1')
            inputs = self.block_layer(inputs=inputs, filters=128, block_fn=self.bottleneck_block, blocks=self.layers[1],
                                      strides=2, train_mode=False, name='block_layer2')
            inputs = self.block_layer(inputs=inputs, filters=256, block_fn=self.bottleneck_block, blocks=self.layers[2],
                                      strides=2, train_mode=False, name='block_layer3')
            inputs = self.block_layer(inputs=inputs, filters=512, block_fn=self.bottleneck_block, blocks=self.layers[3],
                                      strides=2, train_mode=False, name='block_layer4')

            inputs = tf_utils.relu(inputs, name='before_gap_relu', logger=None)
            _, h, w, _ = inputs.get_shape().as_list()
            inputs = tf_utils.avg_pool(inputs, name='gap', ksize=[1, h, w, 1], strides=[1, 1, 1, 1], logger=None)

            inputs = tf_utils.flatten(inputs, name='flatten', logger=None)
            inputs = tf_utils.linear(inputs, 512, name='FC1')
            logits = tf_utils.linear(inputs, self.num_attribute, name='FC2')

            return logits

    def block_layer(self, inputs, filters, block_fn, blocks, strides, train_mode, name):
        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(inputs, filters, train_mode, self.projection_shortcut, strides, name + '_1')

        for num_iter in range(1, blocks):
            inputs = block_fn(inputs, filters, train_mode, None, 1, name=(name + '_' + str(num_iter + 1)))

        return tf.identity(inputs, name)

    def bottleneck_block(self, inputs, filters, train_mode, projection_shortcut, strides, name):
        with tf.compat.v1.variable_scope(name):
            shortcut = inputs
            inputs = tf_utils.relu(inputs, name='relu_0', logger=None)

            # The projection shortcut shouldcome after the first batch norm and ReLU since it perofrms a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = self.projection_shortcut(inputs=inputs, filters_out=filters, strides=strides, name='conv_projection')

            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv_0')
            inputs = tf_utils.relu(inputs, name='relu_1', logger=None)
            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, name='conv_1')

            output = tf.identity(inputs + shortcut, name=(name + '_output'))
            tf_utils.print_activations(output, logger=None)

            return output

    def projection_shortcut(self, inputs, filters_out, strides, name):
        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, name=name)
        return inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, name):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        inputs = tf_utils.conv2d(inputs, output_dim=filters, k_h=kernel_size, k_w=kernel_size,
                                 d_h=strides, d_w=strides, initializer='He', name=name,
                                 padding=('SAME' if strides == 1 else 'VALID'), logger=None)
        return inputs

    @staticmethod
    def fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        inputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]])
        return inputs

    @staticmethod
    def normalize(data):
        return data - 127.5

    @staticmethod
    def preprocessing(left_img, right_img, top_left=(20, 90), bottom_right = (390, 575), resize_factor=0.5):
        # Stage 1: cropping
        left_img_crop = left_img[top_left[0]:bottom_right[0], top_left[1]: bottom_right[1]]
        right_img_crop = right_img[top_left[0]:bottom_right[0], top_left[1]: bottom_right[1]]

        # Stage 2: BGR to Gray
        left_img_gray = cv2.cvtColor(left_img_crop, cv2.COLOR_BGR2GRAY)
        right_img_gray = cv2.cvtColor(right_img_crop, cv2.COLOR_BGR2GRAY)

        # Stage 3: Resize img
        left_img_resize = cv2.resize(left_img_gray, None, fx=resize_factor, fy=resize_factor)
        right_img_resize = cv2.resize(right_img_gray, None, fx=resize_factor, fy=resize_factor)

        # Stage 4: Concatenate left and right img
        input_img = np.dstack([left_img_resize, right_img_resize])

        return input_img

    def _read_min_max_info(self):
        min_max_data = np.load(os.path.join('../data', 'rg_train' + self.data + '.npy'))
        self.x_min = min_max_data[0]
        self.x_max = min_max_data[1]
        self.y_min = min_max_data[2]
        self.y_max = min_max_data[3]
        self.ra_min = min_max_data[4]
        self.ra_max = min_max_data[5]
        self.rb_min = min_max_data[6]
        self.rb_max = min_max_data[7]
        self.f_min = min_max_data[8]
        self.f_max = min_max_data[9]
        self.d_min = min_max_data[10]
        self.d_max = min_max_data[11]

        self.min_values = np.asarray([self.x_min, self.y_min, self.ra_min, self.rb_min, self.f_min, self.d_min])
        self.max_values = np.asarray([self.x_max, self.y_max, self.ra_max, self.rb_max, self.f_max, self.d_max])
        print('Min values: {}'.format(self.min_values))
        print('Max values: {}'.format(self.max_values))

    def load_model(self):
        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # Initialize saver
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(self.model_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            return True, iter_time
        else:
            return False, None


def main(left_path, right_path):
    # Read left and right img
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    # Initialize model
    model = ResNet18()
    pred = model.predict(left_img=left_img, right_img=right_img)

    print('\nPrediction!')
    print('X:  {:.2f}'.format(pred[0, 0]))
    print('Y:  {:.2f}'.format(pred[0, 1]))
    print('Ra: {:.2f}'.format(pred[0, 2]))
    print('Rb: {:.2f}'.format(pred[0, 3]))
    print('F:  {:.2f}'.format(pred[0, 4]))
    print('D:  {:.2f}'.format(pred[0, 5]))


if __name__ == '__main__':
    left_img_path = '../data/rg_train01/A1_L_X-0.500_Y-0.500_Z-0.153_Ra0.000_Rb0.000_F0.610_D0.742.jpg'
    right_img_path = '../data/rg_train01/B1_R_X-0.500_Y-0.500_Z-0.153_Ra0.000_Rb0.000_F0.610_D0.742.jpg'

    main(left_img_path, right_img_path)