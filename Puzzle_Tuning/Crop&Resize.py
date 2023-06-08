import os

import shutil
import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from tqdm import tqdm
import cv2
from torchvision import transforms
from PIL import ImageFile
import pandas as pd

Image.MAX_IMAGE_PIXELS = None


def save_file(f_image, save_dir, suffix='.jpg'):
    """c
    重命名并保存图片，生成重命名的表
    """
    f_image = f_image.convert('RGB')
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)

    """if os.path.exists(file_pack_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(file_pack_path)"""


def find_all_files(root, suffix='tiff'):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
        print(files)
    return res


def data_crop_resize(class_root, output_root = None, size = 384):
    all_data = find_all_files(class_root, 'jpg')
    for data_root in all_data:
        if data_root.endswith('.txt'):
            continue
        if data_root.endswith('.DS_Store'):
            continue
        data_root_index = data_root.find('#1')
        if output_root == None:
            new_data_root = (data_root[:data_root_index+2] + '_Lite' + data_root[data_root_index+2 :]).split('.')[0]
            # specially made for GS dataset:
            """new_data_root = ((data_root[:data_root_index + 2] + '_Lite' + data_root[data_root_index + 2:])).replace('.', '_')
            new_data_root = new_data_root.replace('_jpg', '')"""
        else:
            data_name_without_suffix = os.path.split(data_root)[1].split('.')[0]
            data_class = os.path.split(os.path.split(data_root)[0])[1]
            new_data_root = os.path.join(output_root, data_class)
            new_data_root = os.path.join(new_data_root, data_name_without_suffix)


        img = Image.open(data_root)
        width, height = img.size  # Get dimensions
        a = min(width, height)
        left = int((width - a) / 2)
        top = int((height - a) / 2)
        right = left + a
        bottom = top + a

        # Crop the center of the image
        img = img.crop([left, top, right, bottom])
        resized_img = img.resize((int(size), int(size)), Image.ANTIALIAS)
        save_file(resized_img, new_data_root)


if __name__ == '__main__':
    data_crop_resize(r'D:\CPIA_Version1\PuzzleTuningDatasets#1\S_Scale\WBC_Test-B',
                     r'D:\CPIA_Version1\PuzzleTuningDatasets#1_Lite\S_Scale\WBC_Test-B',
                     384)