'''
@Project ：FlatAngleAPI
@File    ：start.py
@Author  ：谢逸帆
@Date    ：2021/7/14 9:23
'''
import logging
import time
import uuid
import requests
from flask import Flask, request, abort, jsonify
import json
import os


from Segmentation.U2Net import utils_v2
from Segmentation.UNet import utils_v1
from Recognization import utils
#from Enhancement import enhancement_utils

import cv2
from log import *

app = Flask(__name__)

# 服务器保存结果图片的文件夹名
CACHE_FOLDER = "static"

# 服务器保存原图的文件夹名
ORIGINAL_IMAGE_DIR = "before/"

# 启动日志
console_out("logging.log")


@app.route('/segmentation/online/', methods=['POST'])
def segmentation_online():
    """
    第一版本的请求在线图片分割的接口

    :return: 目前暂时只返回结果图像和预测时间
    """
    try:
        # 解析POST请求
        image_name = decode_POST('online_img_url')
        image_path = ORIGINAL_IMAGE_DIR + image_name

        version = request.args.get('version')
        print(version)
        if version == '1':
            # 调用模型进行预测并(处理原图)
            start_time = time.time()

            logging.info('在线图片分割接口v1--准备进行处理')
            file_ds = utils_v1.load_dataset(image_name)
            print(file_ds)
            image_list = [utils_v1.process(file, image_path) for file in file_ds]
            os.remove(image_path)
            logging.info('在线图片分割接口v1--处理成功')

            end_time = time.time()
            cost_time = end_time - start_time

            result_path = ''
            for result in image_list:
                result_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
                cv2.imwrite(result_path, result)
                logging.info(fr'{result_path} 保存结果成功')

            # 服务器本地保存结果
            return jsonify(generate_result(cost_time, result_path))

        elif version == '2':
            start_time = time.time()

            logging.info('在线图片分割接口v2--准备进行处理')
            merged_image, object_scale, img_path_list = utils_v2.process(image_name)
            os.remove(img_path_list[0])
            logging.info('在线图片分割接口v2--处理成功')

            end_time = time.time()
            cost_time = end_time - start_time

            # 服务器本地保存结果
            result_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
            cv2.imwrite(result_path, merged_image)

            return jsonify(generate_result(cost_time, result_path, object_scale))

        else:
            return jsonify(generate_result(0, error='缺少参数或参数不正确'))
    except:
        return jsonify(generate_result(0, error='服务器内部错误，无法处理'))


@app.route('/segmentation/file/', methods=['POST'])
def segmentation_file():
    """
    第一版本的请求图片文件分割的接口

    :return: 目前暂时只返回结果图像和预测时间
    """
    try:
        image_name = decode_POST('image_file')
        image_path = ORIGINAL_IMAGE_DIR + image_name

        version = request.args.get('version')

        if version == '1':
            # 调用模型进行预测并(处理原图)
            start_time = time.time()

            logging.info('图片文件分割接口v1--准备进行处理')
            file_ds = utils_v1.load_dataset(image_name)
            image_list = [utils_v1.process(file, image_path) for file in file_ds]
            os.remove(image_path)

            logging.info('图片文件分割接口v1--处理成功')

            end_time = time.time()
            cost_time = end_time - start_time

            # 服务器本地保存结果
            result_path = ''
            for result in image_list:
                result_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
                cv2.imwrite(result_path, result)
                logging.info(fr'{result_path}保存结果成功')

            return jsonify(generate_result(cost_time, result_path))

        elif version == '2':
            # 调用模型进行预测并(处理原图)
            start_time = time.time()

            logging.info('图片文件分割接口v2--准备进行处理')
            merged_image, object_scale, img_path_list = utils_v2.process(image_name)
            os.remove(img_path_list[0])
            logging.info('图片文件分割接口v2--处理成功')

            end_time = time.time()
            cost_time = end_time - start_time

            result_path = CACHE_FOLDER + '/' + str(uuid.uuid1()) + '.png'
            cv2.imwrite(result_path, merged_image)

            # 服务器本地保存结果
            return jsonify(generate_result(cost_time, result_path, object_scale))
        else:
            return jsonify(generate_result(0, error='缺少参数或参数不正确'))
    except:
        return jsonify(generate_result(0, error='服务器内部错误，无法处理'))


@app.route('/recognization/online/', methods=['POST'])
def recognization_online():
    try:
        image_name = decode_POST('online_img_url')
        image_path = ORIGINAL_IMAGE_DIR + image_name
        start_time = time.time()

        logging.info('在线图片识别接口--准备进行处理')

        file_ds = utils.load_dataset(image_name)
        result = [utils.process(file) for file in file_ds]
        os.remove(image_path)

        logging.info('在线图片识别接口--处理成功')

        end_time = time.time()
        cost_time = end_time - start_time
        return jsonify(generate_result(cost_time, accuracy=result[0][0], label=result[0][1]))
    except:
        return jsonify(generate_result(0, error='服务器内部错误，无法处理'))


@app.route('/recognization/file/', methods=['POST'])
def recognization_file():
    try:
        image_name = decode_POST('image_file')
        image_path = ORIGINAL_IMAGE_DIR + image_name
        start_time = time.time()

        logging.info('图片文件识别接口--准备进行处理')

        file_ds = utils.load_dataset(image_name)
        result = [utils.process(file) for file in file_ds]
        os.remove(image_path)

        logging.info('图片文件识别接口--处理成功')

        end_time = time.time()
        cost_time = end_time - start_time
        return jsonify(generate_result(cost_time, accuracy=result[0][0], label=result[0][1]))

    except:
        return jsonify(generate_result(0, error='服务器内部错误，无法处理'))

#
# @app.route('/enhancement/online/', methods = ['POST'])
# def enhancement_online():
#     try:
#         image_name = decode_POST('online_img_url')
#         image_path  = ORIGINAL_IMAGE_DIR + image_name
#
#         #---------------可能需要判断一下图片的分辨率-------
#
#         #---------------------------------------------
#
#         logging.info('在线图片分辨率增强接口--准备进行处理')
#         start_time = time.time()
#
#         enhancement_utils.process(image_name)
#
#         os.remove(image_path)
#
#         end_time = time.time()
#         cost_time = end_time - start_time
#         logging.info('在线图片分辨率增强接口--处理成功')
#
#         names = image_name.split('.')
#         result_path = CACHE_FOLDER + '/' + names[0] + '-outputs.png'
#         return jsonify(generate_result(cost_time, result_path=result_path))
#     except:
#         abort(500)
#
#
# @app.route('/enhancement/file/', methods = ['POST'])
# def enhancement_file():
#     try:
#         image_name = decode_POST('image_file')
#         image_path = ORIGINAL_IMAGE_DIR + image_name
#
#         # ---------------可能需要判断一下图片的分辨率-------
#
#         # ---------------------------------------------
#
#         logging.info('在线图片分辨率增强接口--准备进行处理')
#         start_time = time.time()
#
#         enhancement_utils.process(image_name)
#
#         os.remove(image_path)
#
#         end_time = time.time()
#         cost_time = end_time - start_time
#         logging.info('在线图片分辨率增强接口--处理成功')
#
#         names = image_name.split('.')
#         result_path = CACHE_FOLDER + '/' + names[0] + '-outputs.png'
#         return jsonify(generate_result(cost_time, result_path=result_path))
#     except:
#         abort(500)


def decode_POST(data_name):
    """
    解析POST请求数据并将图片保存至本地

    :param data_name:POST请求中的数据名
    :return: 保存后的图片名
    """
    if data_name == 'online_img_url':
        data = request.get_data()
        json_data = json.loads(data.decode(("utf-8")))
        image_url = json_data['online_img_url']
        print(image_url)
        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            # 限制为图片文件且图片大小在20MB以下
            content_type = str(r.headers['Content-type'])
            types = content_type.split('/')
            if (types[0] == 'image' and int(r.headers['Content-Length']) > 0 and int(
                    r.headers['Content-Length']) < 20971520):
                logging.info('在线图片分割接口——获得图片数据流')
                img_name = str(uuid.uuid1()) + fr'.{types[1]}'
                with open(ORIGINAL_IMAGE_DIR + img_name, 'wb') as f:
                    f.write(r.content)
                    logging.info('写原图成功')
                return img_name

            #若满足小于20MB图片文件的要求，则返回415
            else:
                return jsonify(generate_result(0, error='图片文件不符合要求'))
        # 若访问图片url失败，则返回404
        else:
            return jsonify(generate_result(0, error='图片url访问失败'))

    elif data_name == 'image_file':
        logging.info('图片文件分割接口——准备获取图片数据流')
        data = request.files.get('image_file')

        content_type = str(data.content_type)
        types = content_type.split('/')
        if data is None:
            return jsonify(generate_result(0, error='图片文件不符合要求'))
        #限制为图片文件
        elif types[0] == 'image':
            logging.info('图片文件分割接口——获得图片数据流')
            img_name = str(uuid.uuid1()) + fr'.{types[1]}'
            data.save(ORIGINAL_IMAGE_DIR + img_name)
            logging.info('写原图成功')
            return img_name

        else:
            return jsonify(generate_result(0, error='图片文件不符合要求'))

    else:
        return jsonify(generate_result(0, error='服务器内部错误，无法处理'))


def generate_result(cost_time, result_path=None, object_scale=None, accuracy=None, label = None, error=None):
    """
    生成返回结果（还需要添加其他参数）

    :param label: 图片对应的类别
    :param accuracy: 识别正确的概率
    :param result_path: 结果文件路径
    :param cost_time: 预测所花费时间
    :param object_scale: 主体覆盖率
    :return: 字典格式的结果
    """

    res = {"cost_time": cost_time}
    if result_path != None:
        res['result'] = "/" + result_path   # 分割结果图片在服务器上的路径
    if object_scale != None:   #主体覆盖率
        res['object_scale'] = fr'{str(round(object_scale * 100, 2))}%'
    if accuracy != None:       #识别正确的概率
        res['accuracy'] = fr'{str(accuracy)}%'
    if label != None:       #图片对应的类别
        res['label'] = fr'{label}'
    if error != None:
        res['error'] = error

    return res


if __name__ == '__main__':
    app.debug = True
    app.run()
