from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim
import os
from lib.model import data_loader, generator, SRGAN, test_data_loader,  save_images
from lib.ops import *
from flag_config import FLAGS
import math
import time
import numpy as np

tf.compat.v1.disable_eager_execution()

print_configuration_op(FLAGS)

if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')


if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)


if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# The testing mode
if FLAGS.mode == 'test':

    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')
    if FLAGS.flip == True:
        FLAGS.flip = False

    if FLAGS.crop_size is not None:
        FLAGS.crop_size = None
    # -----------上述为参数预置和参数错误排除------------

    # -------------读入数据和数据预处理----------------
    test_data = test_data_loader(FLAGS)
    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    with tf.name_scope('convert_image'):

        inputs = deprocessLR(inputs_raw)
        outputs = deprocess(gen_output)

        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope('encode_image'):
        save_fetch = {
            "path_LR": path_LR,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }
    # ---------读入数据和数据预处理-------------

    # ----------权重加载------------

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)  # 该方法第二个参数为权重路径
    # ---------------权重加载----------------

    # ----------------图片处理---------------
        max_iter = len(test_data.inputs)
        print('Evaluation starts!!')
        for i in range(max_iter):
            input_im = np.array([test_data.inputs[i]]).astype(np.float32)
            path_lr = test_data.paths_LR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im,
                                                      path_LR: path_lr})
            filesets = save_images(results, FLAGS)
            for i, f in enumerate(filesets):
                print('evaluate image', f['name'])





