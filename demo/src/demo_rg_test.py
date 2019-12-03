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
    def __init__(self, input_shape=(185, 208), num_attribute=6, mode=1, domain='xy', name='ResNet18'):
        self.input_shape = (*input_shape, 2 if mode == 0 else 1)
        self.num_attribute = num_attribute
        self.mode = mode
        self.domain = domain
        self.name = name
        self.layers = [2, 2, 2, 2]
        self.small_value = 1e-7
        self.sess = tf.compat.v1.Session()  # Initialize session

        # Model should be fixed in here
        if self.mode == 1 and self.domain == 'xy':
            self.model_dir = '../model/20191114-210545'
        elif self.mode == 1 and self.domain == 'rarb':
            self.model_dir = '../model/20191114-110556'
        self.data = '01'

        self._read_min_max_info()
        self._build_graph()

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
        self.preds = self.forward_network(input_img=self.normalize_img(self.img_tfph), reuse=False)
        self.unnorm_preds = self.unnormalize(self.preds)

    def predict(self, left_img=None, right_img=None):
        # Preprocessing
        pre_img = self.preprocessing(left_img, right_img)
        output = self.sess.run(self.unnorm_preds, feed_dict={self.img_tfph: np.expand_dims(pre_img, axis=0)})
        return output[0]

    def forward_network(self, input_img, reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            tf_utils.print_activations(input_img, logger=None)
            inputs = self.conv2d_fixed_padding(inputs=input_img, filters=64, kernel_size=7, strides=1, name='conv1')
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

            inputs = tf_utils.relu(inputs, name='before_flatten_relu', logger=None)

            # Flatten & FC1
            inputs = tf_utils.flatten(inputs, name='flatten', logger=None)
            inputs = tf_utils.linear(inputs, 512, name='FC1')
            inputs = tf_utils.relu(inputs, name='FC1_relu', logger=None)

            inputs = tf_utils.linear(inputs, 256, name='FC2')
            inputs = tf_utils.relu(inputs, name='FC2_relu', logger=None)

            logits = tf_utils.linear(inputs, self.num_attribute, name='Out')
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
    def normalize_img(data):
        return (data - 127.5) / 127.5

    def unnormalize(self, data):
        return tf.maximum(data, 0.) * (self.max_values - self.min_values + self.small_value) + self.min_values

    @staticmethod
    def preprocessing(left_img=None, right_img=None, top_left=(20, 100), bottom_right=(390, 515), resize_factor=0.5,
                      binarize_threshold=55.):
        imgs = list()

        for img in [left_img, right_img]:
            if img is not None:
                # Stage 1: cropping
                img_crop = img[top_left[0]:bottom_right[0], top_left[1]: bottom_right[1]]
                # Stage 2: BGR to Gray
                img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                # Stage 3: Thresholding
                _, img_binary = cv2.threshold(img_gray, binarize_threshold, 255., cv2.THRESH_BINARY)
                # Stage 4: Resize img
                img_resize = cv2.resize(img_binary, None, fx=resize_factor, fy=resize_factor,
                                        interpolation=cv2.INTER_NEAREST)
                imgs.append(img_resize)

        if len(imgs) == 2:
            # Concatenate left and right img
            input_img = np.dstack([imgs[0], imgs[1]])
        else:
            input_img = np.expand_dims(imgs[0], axis=-1)

        print('input_img shape: {}'.format(input_img.shape))

        return input_img

    def _read_min_max_info(self):
        min_max_data = np.load(os.path.join('../data', 'rg_' + self.domain + '_train_' + self.data + '.npy'))
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

    ####################################################################################################################
    # README!
    # mode=0: left and right image
    # mode=1: left image only
    # mode=2: right image only
    # domain='xy': xy prediction model
    # domain='rarb': rarb prediction model
    ####################################################################################################################

    # Initialize model
    model = ResNet18(mode=1, domain='xy')
    pred = model.predict(left_img=left_img, right_img=None)

    print('\nPrediction!')
    print('X:  {:.3f}'.format(pred[0]))
    print('Y:  {:.3f}'.format(pred[1]))
    print('Ra: {:.3f}'.format(pred[2]))
    print('Rb: {:.3f}'.format(pred[3]))
    print('F:  {:.3f}'.format(pred[4]))
    print('D:  {:.3f}'.format(pred[5]))


if __name__ == '__main__':
    left_img_path = '../data/rg_xy_train_01/A5_L_X-0.500_Y0.500_Z-1.054_Ra0.000_Rb0.000_F0.130_D0.049.jpg'
    right_img_path = '../data/rg_xy_train_01/B5_R_X-0.500_Y0.500_Z-1.054_Ra0.000_Rb0.000_F0.130_D0.049.jpg'

    main(left_img_path, right_img_path)