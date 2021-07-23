'''
该文件用于搭建模型，加载预训练好的权重，对请求中的图片进行预处理以及识别预测

@Project ：flatangleAPI 
@File    ：utils.py
@Author  ：谢逸帆
@Date    ：2021/7/20 14:48 
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from Recognization.info import *
import numpy as np
import math
from tensorflow.keras import layers, models
import pickle

# 搭建识别模型
model = models.Sequential()
model.add(layers.Input((MODEL_SIZE, MODEL_SIZE, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(LABEL_NUM))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.load_weights(r'D:/PyCharm 2021.1.3/PycharmProjects/flatangleAPI/Recognization/weights.h5')

# 原图存放路径
file_root = WORK_SPACE

# 加载记录了index对应的类别的文件（服务器上需要修改路径）
index_to_label = pickle.load(open(r'D:/PyCharm 2021.1.3/PycharmProjects/flatangleAPI/Recognization/index_to_label.byte', 'rb'))

# MIN为绝对正确的界限（大于等于0），MAX为绝对错误的界限
# @author=戴柯
VALUE_MIN = 0
VALUE_MAX = 60

# 曲线更陡峭，对错误更敏感（使用导数单调递增的函数时，使用线性函数则无意义）
# （注意使用非单调、导数非单调函数时的OFFSET的设置应保证在范围（VALUE_MAX, VALUE_MIN）内单调
# @author=戴柯
OFFSET = 0


def load_and_decode_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (MODEL_SIZE, MODEL_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_dataset(image_name):
    """
    将客户端传来的图片文件转为张量数据集
    """
    file_paths = list(pathlib.Path(file_root).glob(fr'{image_name}'))
    file_paths = list(str(path) for path in file_paths)
    file_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    file_ds = file_ds.map(load_and_decode_jpg)
    file_ds = file_ds.batch(1)
    return file_ds


def fun(num):
    num += OFFSET
    return pow(num, 2)


def process(file):
    """
    调用模型进行识别预测
    :param file: 需要预测的图片张量
    :return: 最大可能的正确率，最大可能的类别
    """
    result = model.predict(file)
    buf = 0
    index = 0
    index_buf = -1
    for num in result[0]:
        if num > buf:
            buf = num
            index_buf = index
        index += 1
    buf = 0
    for num in result[0]:
        buf += num
    buf /= len(index_to_label)
    value = result[0][index_buf] - buf
    value = (
            (fun(value) - fun(VALUE_MIN))
            /
            (fun(VALUE_MAX) - fun(VALUE_MIN))
            )
    accuracy = int(value*100)
    label = index_to_label[int(index_buf)]
    return accuracy, label
