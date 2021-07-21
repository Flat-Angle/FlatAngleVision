"""

    @version: 2.0
    @Author	: 肖思宇

"""

import cv2
import requests as requests
from flask import Flask, request, abort, jsonify, render_template, url_for, make_response
import os.path
import time
import uuid
import json

from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'NMSL'}), 500)


@app.route('/FlatAngle')
def toFlatAngle():
    return render_template('index.html')


@app.route('/segmentation_upload', methods=['POST', 'GET'])
def toSegmentationUpload():
    download_path = ""
    imageUrl = ""
    print("This is test1")

    if request.method == 'POST':
        imageUrl = request.form.get('online_img_url')
        file = request.files.get('')

        if imageUrl != "":
            print("this is online image")
            onlineUrl = "http://121.199.2.56:5000/segmentation/online"
            data = {'online_img_url': imageUrl}
            print(imageUrl)
            # print(str(imageUrl))
            result1 = requests.post(onlineUrl, data=json.dumps(data))

            print(result1.text)
            json_dict1 = json.loads(result1.text)

            print(json_dict1["result"])

            download_path = os.path.join(json_dict1["result"])

        else:
            print("this is local image")
            uploadUrl = "http://121.199.2.56:5000/segmentation/upload_file"
            file = request.files.get("image_file")
            basePath = os.path.dirname(__file__)  # 当前文件所在路径

            upload_path = os.path.join(basePath, 'static/images', secure_filename(file.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
            file.save(upload_path)

            # 使用Opencv转换一下图片格式和名称
            img = cv2.imread(upload_path)
            cv2.imwrite(os.path.join(basePath, 'static/images', 'input.png'), img)

            localPath = os.path.join(basePath, 'static/images', 'input.png')
            print(localPath)
            files = {'image_file': open(localPath, 'rb')}

            result2 = requests.post(uploadUrl, files=files)
            print(result2.text)
            json_dict2 = json.loads(result2.text)
            print(json_dict2["result"])
            download_path = os.path.join(json_dict2["result"])
            imageUrl = localPath

        return render_template('pages/segmentation_upload.html', path=download_path, url=imageUrl)

    return render_template('pages/upload.html')


@app.route('/identification_upload', methods=['POST', 'GET'])
def toIdentificationUpload():
    download_path = ""
    imageUrl = ""

    return render_template('pages/identification_upload.html', path=download_path, url=imageUrl)


@app.route('/resolution_upload', methods=['POST', 'GET'])
def toResolutionUpload():
    download_path = ""
    imageUrl = ""
    print("This is test1")

    if request.method == 'POST':
        imageUrl = request.form.get('online_img_url')
        file = request.files.get('')

        if imageUrl != "":
            print("this is online image")
            onlineUrl = "http://121.199.2.56:5000/segmentation/online"
            data = {'online_img_url': imageUrl}
            print(imageUrl)
            # print(str(imageUrl))
            result1 = requests.post(onlineUrl, data=json.dumps(data))

            print(result1.text)
            json_dict1 = json.loads(result1.text)

            print(json_dict1["result"])

            download_path = os.path.join(json_dict1["result"])

        else:
            print("this is local image")
            uploadUrl = "http://121.199.2.56:5000/segmentation/upload_file"
            file = request.files.get("image_file")
            basePath = os.path.dirname(__file__)  # 当前文件所在路径
            # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
            upload_path = os.path.join(basePath, 'static/images', secure_filename(file.filename))
            file.save(upload_path)

            # 使用Opencv转换一下图片格式和名称
            img = cv2.imread(upload_path)
            cv2.imwrite(os.path.join(basePath, 'static/images', 'input.png'), img)

            localPath = os.path.join(basePath, 'static/images', 'input.png')
            print(localPath)
            files = {'image_file': open(localPath, 'rb')}

            result2 = requests.post(uploadUrl, files=files)
            print(result2.text)
            json_dict2 = json.loads(result2.text)
            print(json_dict2["result"])
            download_path = os.path.join(json_dict2["result"])
            imageUrl = localPath

        return render_template('pages/segmentation_upload.html', path=download_path, url=imageUrl)

    return render_template('pages/upload.html')


@app.route('/about')
def toAboutUs():
    download_path = ""
    imageUrl = ""

    return render_template('pages/AboutUs.html', path=download_path, url=imageUrl)


if __name__ == '__main__':
    app.run()
