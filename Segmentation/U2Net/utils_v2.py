'''
该文件用于加载U2Net模型，对请求中的图片进行预处理以及预测
@Project ：flatangleAPI 
@File    ：utils_v2.py
@Author  ：谢逸帆
@Date    ：2021/7/19 14:40 
'''
import os
from collections import Counter
import cv2
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from Segmentation.U2Net.data_loader import RescaleT
from Segmentation.U2Net.data_loader import ToTensor
from Segmentation.U2Net.data_loader import ToTensorLab
from Segmentation.U2Net.data_loader import SalObjDataset
from Segmentation.U2Net.model import U2NET

# 加载U2NET模型
model_name = 'u2net'
net = U2NET(3, 1)

# 模型权重的路径
model_dir = r"D:\PyCharm 2021.1.3\PycharmProjects\flatangleAPI\Segmentation\U2Net\saved_models\u2net\u2net.pth"
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
net.eval()


# 服务器保存原图的文件夹名
image_dir = r"D:\PyCharm 2021.1.3\PycharmProjects\flatangleAPI\before"

# 结果保存的文件夹名
prediction_dir = r"D:\PyCharm 2021.1.3\PycharmProjects\flatangleAPI\static"


def normPRED(d):
    """
    图片标准化
    """
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)
    return dn


def actual_process(image_path, pred):
    """
    真正处理的函数

    :param image_path:原图文件路径
    :return: 原图和主体覆盖率
    """
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    # offset 用于定义切割主体的敏感度，越低则越敏感，取值于1.0~2.0
    offset = 1.8
    # -------------process the image---------------------
    image = cv2.imread(image_path)
    mask_pre = (predict_np * offset).astype(np.int16)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    mask = np.array(mask_pre)
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 4, axis=2)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # # ----------- 优化部分:若原图过大，导致结果图片锯齿严重，增加中值滤波和高斯滤波 ------------------
    if image.shape[0] >= 640 and image.shape[1] >= 640:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, 1)
        mask = cv2.medianBlur(mask, 5)
    # --------------------------------------
    # merged_image:切割后的目标图片
    merged_image = np.multiply(mask, image)
    # object_scale:目标主体在图片中所占的比例
    mask_flat = mask.flatten()
    num_object_pix = Counter(mask_flat)[1]
    object_scale = num_object_pix/len(mask_flat)
    return merged_image, object_scale


def process(image_name):
    """
    封装处理的方法

    :return:原图和主体覆盖率
    """
    img_path_list = glob.glob(image_dir + os.sep + fr'{image_name}')

    test_salobj_dataset = SalObjDataset(img_name_list=img_path_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0)

    for i_test, data_test in enumerate(test_salobj_dataloader):

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        merged_image, object_scale = actual_process(img_path_list[i_test], pred)
        del d1, d2, d3, d4, d5, d6, d7
        return merged_image, object_scale, img_path_list




