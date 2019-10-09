# ---------------------------------------------------------
# Tensorflow Utils Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


def padding2d(x, p_h=1, p_w=1, pad_type='REFLECT', name='pad2d'):
    if pad_type == 'REFLECT':
        return tf.pad(x, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], 'REFLECT', name=name)


def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, initializer=None, padding='SAME', name='conv2d',
           is_print=True, logger=None):
    with tf.compat.v1.variable_scope(name):
        if initializer is None:
            init_op = tf.truncated_normal_initializer(stddev=stddev)
        elif initializer.lower() == 'he':
            init_op = tf.compat.v1.initializers.he_normal()
        elif initializer.lower() == 'xavier':
            init_op = tf.contrib.layers.xavier_initializer()
        else:
            raise NotImplementedError

        w = tf.compat.v1.get_variable('w', [k_h, k_w, x.get_shape()[-1], output_dim], initializer=init_op)
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.compat.v1.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases, name='add')

        if is_print:
            print_activations(conv, logger)

        return conv


def deconv2d(x, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, initializer=None, padding_='SAME',
             output_size=None, name='deconv2d', with_w=False, is_print=True, logger=None):
    with tf.compat.v1.variable_scope(name):
        input_shape = x.get_shape().as_list()

        # calculate output size
        h_output, w_output = None, None
        if not output_size:
            h_output, w_output = input_shape[1] * 2, input_shape[2] * 2
        # output_shape = [input_shape[0], h_output, w_output, k]  # error when not define batch_size
        output_shape = [tf.shape(x)[0], h_output, w_output, output_dim]

        # conv2d transpose
        if initializer is None:
            init_op = tf.random_normal_initializer(stddev=stddev)
        elif initializer.lower() == 'he':
            init_op = tf.compat.v1.initializers.he_normal()
        elif initializer.lower() == 'xavier':
            init_op = tf.contrib.layers.xavier_initializer()
        else:
            raise NotImplementedError
        w = tf.compat.v1.get_variable('w', [k_h, k_w, output_dim, input_shape[3]], initializer=init_op)
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding=padding_)

        biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if is_print:
            print_activations(deconv, logger)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def concat(values, axis, name='concat', is_print=True, logger=None):
    output = tf.concat(values=values, axis=axis, name=name)

    if is_print:
        print_activations(output, logger)

    return output


def upsampling2d(x, size=(2, 2), name='upsampling2d'):
    with tf.name_scope(name):
        shape = x.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(x, size=(size[0] * shape[1], size[1] * shape[2]))

def flatten(x, name='flatten', data_format='channels_last', is_print=True, logger=None):
    try:
        output = tf.layers.flatten(inputs=x, name=name, data_format=data_format)
    except(RuntimeError, TypeError, NameError):
        print('[*] Catch the flatten function Error!')
        output = tf.contrib.layers.flatten(inputs=x, scope=name)

    if is_print:
        print_activations(output, logger)
    return output


def linear(x, output_size, bias_start=0.0, stddev=0.02, initializer=None, name='fc', with_w=False,
           is_print=True, logger=None):
    shape = x.get_shape().as_list()

    with tf.compat.v1.variable_scope(name):
        if initializer is None:
            init_op = tf.truncated_normal_initializer(stddev=stddev)
        elif initializer.lower() == 'he':
            init_op = tf.initializers.he_normal()
        elif initializer.lower() == 'xavier':
            init_op = tf.contrib.layers.xavier_initializer()
        else:
            raise NotImplementedError

        matrix = tf.compat.v1.get_variable(name="matrix", shape=[shape[1], output_size],
                                           dtype=tf.float32, initializer=init_op)
        bias = tf.compat.v1.get_variable(name="bias", shape=[output_size],
                                         initializer=tf.compat.v1.constant_initializer(bias_start))
        output = tf.matmul(x, matrix) + bias

        if is_print:
            print_activations(output, logger)

        if with_w:
            return output, matrix, bias
        else:
            return output


def norm(x, name, _type, _ops, is_train=True, is_print=True, logger=None):
    if _type == 'batch':
        return batch_norm(x, name=name, _ops=_ops, is_train=is_train, is_print=is_print, logger=logger)
    elif _type == 'instance':
        return instance_norm(x, name=name, is_print=is_print, logger=logger)
    else:
        raise NotImplementedError


def batch_norm(x, name, _ops, is_train=True, is_print=True, logger=None):
    """Batch normalization."""
    with tf.compat.v1.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.compat.v1.get_variable('beta', params_shape, tf.float32,
                               initializer=tf.compat.v1.constant_initializer(0.0))
        gamma = tf.compat.v1.get_variable('gamma', params_shape, tf.float32,
                                initializer=tf.compat.v1.constant_initializer(1.0))

        if is_train is True:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.compat.v1.get_variable('moving_mean', params_shape, tf.float32,
                                          initializer=tf.compat.v1.constant_initializer(0.0),
                                          trainable=False)
            moving_variance = tf.compat.v1.get_variable('moving_variance', params_shape, tf.float32,
                                              initializer=tf.compat.v1.constant_initializer(1.0),
                                              trainable=False)

            _ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
            _ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
        else:
            mean = tf.compat.v1.get_variable('moving_mean', params_shape, tf.float32,
                initializer=tf.compat.v1.constant_initializer(0.0), trainable=False)
            variance = tf.compat.v1.get_variable('moving_variance', params_shape, tf.float32,
                initializer=tf.compat.v1.constant_initializer(1.0), trainable=False)

        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)
        y.set_shape(x.get_shape())

        if is_print:
            print_activations(y, logger=logger)

        return y


def instance_norm(x, name='instance_norm', mean=1.0, stddev=0.02, epsilon=1e-5, is_print=True, logger=None):
    with tf.compat.v1.variable_scope(name):
        depth = x.get_shape()[3]
        scale = tf.compat.v1.get_variable(
            'scale', [depth], tf.float32,
            #initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32)
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        offset = tf.compat.v1.get_variable('offset', [depth], initializer=tf.compat.v1.constant_initializer(0.0))

        # calcualte mean and variance as instance
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)

        # normalization
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv

        output = scale * normalized + offset

        if is_print:
            print_activations(output, logger=logger)

        return output


def n_res_blocks(x, _ops=None, norm_='instance', is_train=True, num_blocks=6, is_print=True, logger=None):
    output = None
    for idx in range(1, num_blocks+1):
        output = res_block(x, x.get_shape()[3], _ops=_ops, norm_=norm_, is_train=is_train,
                           name='res{}'.format(idx))
        x = output

    if is_print:
        print_activations(output, logger)

    return output


# norm(x, name, _type, _ops, is_train=True)
def res_block(x, k, _ops=None, norm_='instance', is_train=True, pad_type=None, name=None):
    with tf.variable_scope(name):
        conv1, conv2 = None, None

        # 3x3 Conv-Batch-Relu S1
        with tf.variable_scope('layer1'):
            if pad_type is None:
                conv1 = conv2d(x, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv')
            elif pad_type == 'REFLECT':
                padded1 = padding2d(x, p_h=1, p_w=1, pad_type='REFLECT', name='padding')
                conv1 = conv2d(padded1, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID', name='conv')
            normalized1 = norm(conv1, name='norm', _type=norm_, _ops=_ops, is_train=is_train)
            relu1 = tf.nn.relu(normalized1)

        # 3x3 Conv-Batch S1
        with tf.variable_scope('layer2'):
            if pad_type is None:
                conv2 = conv2d(relu1, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv')
            elif pad_type == 'REFLECT':
                padded2 = padding2d(relu1, p_h=1, p_w=1, pad_type='REFLECT', name='padding')
                conv2 = conv2d(padded2, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID', name='conv')
            normalized2 = norm(conv2, name='norm', _type=norm_, _ops=_ops, is_train=is_train)

    # sum layer1 and layer2
    output = x + normalized2
    return output


def res_block_v2(x, k, filter_size, _ops=None, norm_='instance', is_train=True, resample=None, name=None,
                 is_print=True, logger=None):
    with tf.variable_scope(name):
        if resample == 'down':
            conv_shortcut = functools.partial(avgPoolConv, output_dim=k, filter_size=1)
            conv_1 = functools.partial(conv2d, output_dim=k, k_h=filter_size, k_w=filter_size, d_h=1, d_w=1)
            conv_2 = functools.partial(convAvgPool, output_dim=k)
        elif resample == 'up':
            conv_shortcut = functools.partial(deconv2d, k=k)
            conv_1 = functools.partial(deconv2d, k=k, k_h=filter_size, k_w=filter_size)
            conv_2 = functools.partial(conv2d, output_dim=k, k_h=filter_size, k_w=filter_size, d_h=1, d_w=1)
        elif resample is None:
            conv_shortcut = functools.partial(conv2d, output_dim=k, k_h=filter_size, k_w=filter_size, d_h=1, d_w=1)
            conv_1 = functools.partial(conv2d, output_dim=k, k_h=filter_size, k_w=filter_size, d_h=1, d_w=1)
            conv_2 = functools.partial(conv2d, output_dim=k, k_h=filter_size, k_w=filter_size, d_h=1, d_w=1)
        else:
            raise Exception('invalid resample value')

        if (k == x.get_shape().as_list()[3]) and (resample is None):
            shortcut = x  # Identity skip-connection
        else:
            shortcut = conv_shortcut(x, name='shortcut')

        output = x
        output = norm(output, _type=norm_, _ops=_ops, is_train=is_train, name='norm1', is_print=is_print, logger=logger)
        output = relu(output, name='relu1', is_print=is_print, logger=logger)
        output = conv_1(output, name='conv1', is_print=is_print, logger=logger)
        output = norm(output, _type=norm_, _ops=_ops, is_train=is_train, name='norm2', is_print=is_print, logger=logger)
        output = relu(output, name='relu2', is_print=is_print, logger=logger)
        output = conv_2(output, name='conv2', is_print=is_print, logger=logger)

        return shortcut + output


def convAvgPool(x, output_dim, filter_size=3, stride=1, name='convAvgPool', is_print=True, logger=None):
    with tf.variable_scope(name):
        output = conv2d(x, output_dim=output_dim, k_h=filter_size, k_w=filter_size, d_h=stride, d_w=stride,
                        is_print=is_print, logger=logger)
        output = avg_pool(output, name='avg_pool', is_print=is_print, logger=logger)

        if is_print:
            print_activations(output, logger)

        return output


def avgPoolConv(x, output_dim, filter_size=3, stride=1, name='avgPoolConv', is_print=True, logger=None):
    with tf.variable_scope(name):
        output = avg_pool(x, name='avg_pool', is_print=is_print, logger=logger)
        output = conv2d(output, output_dim=output_dim, k_h=filter_size, k_w=filter_size, d_h=stride, d_w=stride,
                        is_print=True, logger=logger)
        if is_print:
            print_activations(output, logger)

        return output


def identity(x, name='identity', is_print=True, logger=None):
    output = tf.identity(x, name=name)
    if is_print:
        print_activations(output, logger)

    return output


def max_pool(x, name='max_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], is_print=True, logger=None):
    output = tf.nn.max_pool2d(x, ksize=ksize, strides=strides, padding='SAME', name=name)
    if is_print:
        print_activations(output, logger)

    return output


def avg_pool(x, name='avg_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], is_print=True, logger=None):
    output = tf.nn.avg_pool2d(x, ksize=ksize, strides=strides, padding='VALID', name=name)
    if is_print:
        print_activations(output, logger)

    return output


def dropout(x, keep_prob=0.5, seed=None, name='dropout', is_print=True, logger=None):
    try:
        output = tf.nn.dropout(x=x,
                               rate=keep_prob,
                               seed=tf.set_random_seed(seed) if seed else None,
                               name=name)
    except(RuntimeError, TypeError, NameError):
        print('[*] Catch the dropout function Error!')
        output = tf.nn.dropout(x=x,
                               keep_prob=keep_prob,
                               seed=tf.set_random_seed(seed) if seed else None,
                               name=name)

    if is_print:
        print_activations(output, logger)

    return output

def sigmoid(x, name='sigmoid', is_print=True, logger=None):
    output = tf.nn.sigmoid(x, name=name)
    if is_print:
        print_activations(output, logger)

    return output


def tanh(x, name='tanh', is_print=True, logger=None):
    output = tf.nn.tanh(x, name=name)
    if is_print:
        print_activations(output, logger)

    return output


def relu(x, name='relu', is_print=False, logger=None):
    output = tf.nn.relu(x, name=name)
    if is_print:
        print_activations(output, logger)

    return output


def lrelu(x, leak=0.2, name='lrelu', is_print=True, logger=None):
    output = tf.maximum(x, leak*x, name=name)
    if is_print:
        print_activations(output, logger)

    return output


def elu(x, name='elu', is_print=True, logger=None):
    output = tf.nn.elu(x, name=name)
    if is_print:
        print_activations(output, logger)

    return output


def xavier_init(in_dim):
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return xavier_stddev


def print_activations(t, logger=None):
    if logger is None:
        print(t.op.name, ': {}'.format(t.get_shape().as_list()))
    else:
        logger.info(t.op.name + ': {}'.format(t.get_shape().as_list()))


def show_all_variables(logger=None, scope=None):
    total_count = 0

    if scope is None:
        variables = tf.compat.v1.trainable_variables()
    else:
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    for idx, op in enumerate(variables):
        shape = op.get_shape()
        count = np.prod(shape)

        if logger is None:
            print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
        else:
            logger.info("[%2d] %s %s = %s" % (idx, op.name, shape, count))

        total_count += int(count)

    if logger is None:
        print("[Total] variable size: %s" % "{:,}".format(total_count))
    else:
        logger.info("[Total] variable size: %s" % "{:,}".format(total_count))


def batch_convert2int(images):
    # images: 4D float tensor (batch_size, image_size, image_size, depth)
    return tf.map_fn(convert2int, images, dtype=tf.uint8)


def convert2int(image):
    # transform from float tensor ([-1.,1.]) to int image ([0,255])
    return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)
