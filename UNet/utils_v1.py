'''
该文件用于将模型加载，对请求中的图片进行预处理以及预测
@Project ：FlatAngleAPI
@File    ：utils_v1.py
@Author  ：谢逸帆
@Date    ：2021/7/14 14:23 
'''
import cv2
import tensorflow as tf
import pathlib
from Segmentation.UNet.info import *
import numpy as np

file_root = UN_PRODUCED_FILE_PTAH

model = tf.keras.models.load_model(MODULE_PATH)


def load_dataset(image_name):
    """
    将客户端传来的图片文件转为张量数据集
    """
    file_paths = list(pathlib.Path(file_root).glob(fr'{image_name}'))
    file_paths = [str(path) for path in file_paths]
    file_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    file_ds = file_ds.map(load_and_decode_img)
    file_ds = file_ds.batch(1)
    return file_ds


def load_and_decode_img(path):
    """
    根据图片路径将图片转换为张量
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    return img


def process(file, image_path):
    """
    模型进行预测并处理图像
    """
    img = tf.image.resize(file, (MODULE_SIZE, MODULE_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    mask = create_mask(model.predict(img))
    mask = tf.image.resize(mask, (file.shape[1], file.shape[2]))
    mask = mask.numpy()
    # 优化部分
    kernel = np.ones((int(2 * mask.shape[0] / MODULE_SIZE), int(2 * mask.shape[1] / MODULE_SIZE)),
                      dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, 1)
    mask = cv2.medianBlur(mask, 5)
    #
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    mask = np.expand_dims(mask, axis=2)
    image = image * mask

    return image


def create_mask(pred_mask):
    """
     将predict返回的标签掩码图张量译码为012的格式
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = pred_mask[0]

    return pred_mask
