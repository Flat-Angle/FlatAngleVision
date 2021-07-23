'''
@Project ：FlatAngleAPI
@File    ：test.py
@Author  ：谢逸帆
@Date    ：2021/7/14 15:03 
'''
import glob
import os

import requests
import uuid
import cv2
#from Segmentation.UNet import utils_v1
#from Segmentation.U2Net import utils_v2
#from Recognization import utils
from Enhancement import enhancement_utils
CACHE_FOLDER = "static"

# "https://inews.gtimg.com/newsapp_bt/0/7396554241/1000"
# "https://pic2.zhimg.com/80/v2-f6b1f64a098b891b4ea1e3104b5b71f6_720w.png"
# image_url = r"https://inews.gtimg.com/newsapp_bt/0/7396554241/1000"
# r = requests.get(image_url, stream=True)
# print(r.status_code)
# if r.status_code == 200:
#     print(r.headers['Content-Type'])
#     print(r.headers['Content-Length'])  #比特数
#     content_type = str(r.headers['Content-type'])
#     types = content_type.split('/')
#     print(types[0])   #image
#     print(types[1])   #jpeg/png/jpg
#
#     #只有图片url且图片大小小于20MB
#     if(types[0] == 'image' and int(r.headers['Content-Length']) > 0 and int(r.headers['Content-Length']) < 20971520):
#         img_name = str(uuid.uuid1()) + fr'.{types[1]}'
#         img_path = "./Segmentation/before/" + img_name
#         with open('./Segmentation/before/'+img_name, 'wb') as f:
#             f.write(r.content)
#         file_ds = utils_v1.load_dataset(img_name)
#         result_list = [utils_v1.process(file, img_path) for file in file_ds]
#
#         for mask in result_list:
#             mask_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
#             cv2.imwrite(mask_path, mask)

# image_url = r"https://tse3-mm.cn.bing.net/th/id/OIP-C.Dx0-3dtHwn_dpvyfvN3_YwHaKe?w=206&h=291&c=7&o=5&dpr=1.25&pid=1.7"
# r = requests.get(image_url, stream=True)
#
# print(r.status_code)
# if r.status_code == 200:
#     content_type = str(r.headers['Content-type'])
#     types = content_type.split('/')
#     img_name = str(uuid.uuid1())+fr'.{types[1]}'
#     img_path = './Recognization/workspace/' + img_name
#     with open('./Recognization/workspace/' + img_name, 'wb') as f:
#         f.write(r.content)
#
#     file_ds = utils.load_dataset(img_name)
#     result_list = [utils.process(file) for file in file_ds]
#     print(result_list)
#     print(result_list[0][0])
#     print(result_list[0][1])
    # os.remove(img_path_list[0])
    # result_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
    # cv2.imwrite(result_path, merged_image)
    # print('主体覆盖率：'+str(object_scale))

#测试搜索指定图片
# input_path='D:/PyCharm 2021.1.3/PycharmProjects/flatangleAPI/before/'
# image_list_LR_temp = os.listdir(input_path)
# image_name = '000000000139.jpg'
# image_list_LR = [os.path.join(input_path, _) for _ in image_list_LR_temp if _.split('.')[-2] in image_name]
# print(image_list_LR)

image_url = r"https://pic2.zhimg.com/v2-bfc0c011f113529e32cfca3ce554e914_r.jpg"
r = requests.get(image_url, stream=True)

print(r.status_code)
if r.status_code == 200:
    content_type = str(r.headers['Content-type'])
    types = content_type.split('/')
    img_name = str(uuid.uuid1()) + fr'.{types[1]}'
    img_path = './before/' + img_name
    with open('./before/' + img_name, 'wb') as f:
        f.write(r.content)

img_name = '9da76b86-e9ce-11eb-b782-907841dabfac.jpeg'
enhancement_utils.process(img_name)

