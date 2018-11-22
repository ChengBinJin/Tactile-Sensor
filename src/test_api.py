# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import scipy.misc
import time
import numpy as np
import tensorflow as tf

from test_api_class import Model


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'train01', 'dataset name [train01, train02, train03], default: train01')


def all_files_under(path, extension=None, append_path=True, prefix=None, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if prefix in fname]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)
                         if (fname.endswith(extension)) and (prefix in fname)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if prefix in fname]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)
                         if (fname.endswith(extension)) and (prefix in fname)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def load_data(img_path):
    img = scipy.misc.imread(img_path, flatten=True).astype(np.float)
    img = scipy.misc.imresize(img, (240, 320))

    img = img[24:216, 47:271]  # set ROI region
    img = img / 127.5 - 1.0  # scaling and zero centring, output range [-1., 1.]

    # hope output should be [h, w, c]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    return img


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    # Read left and right img paths
    left_img_paths = all_files_under(os.path.join('../data', 'test'+FLAGS.dataset[-2:]), extension='.bmp',
                                     prefix='left')
    right_img_paths = all_files_under(os.path.join('../data', 'test'+FLAGS.dataset[-2:]), extension='.bmp',
                                      prefix='right')
    num_imgs = len(left_img_paths)

    # Initialize model
    model = Model(FLAGS)  # initialize model, it will take about 5 secs.

    total_pt = 0.
    for idx in range(num_imgs):
        # Read img and do preprocessing for the img
        left_img = load_data(left_img_paths[idx])
        right_img = load_data(right_img_paths[idx])
        print('Left img path: {}'.format(left_img_paths[idx]))

        tic = time.time()
        preds = model.predict(left_img, right_img)
        toc = time.time() - tic
        total_pt += toc
        print('Predicts: {}\n'.format(preds))

    print('Avg. PT: {} ms.'.format(total_pt / num_imgs * 1000))


if __name__ == '__main__':
    tf.app.run()
