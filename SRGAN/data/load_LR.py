import cv2
import os
import numpy as np

global file_number
file_number = 0
output_path = './RAISE_LR'
input_path = './RAISE_HR'


def get_file_name(file_dir):

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg' or '.png':
                global file_number
                file_number += 1
                print(file_number)
                pic_HR = cv2.imread(os.path.join(root, file))
                pic_LR = cv2.resize(pic_HR, (int(pic_HR.shape[1] / 4), int(pic_HR.shape[0] / 4)))
                cv2.imwrite(os.path.join(output_path, file), pic_LR)


get_file_name(input_path)
