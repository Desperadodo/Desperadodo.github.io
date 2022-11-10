import os
import re
import csv
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms


def del_file(filepath):
    """
    Delete all files and folders in one directory
    :param filepath: file path
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)
    # del_file(file_pack_path)


def find_all_files(root, suffix=None):
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


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)

def majority_voting(data_dir, gleason_type, mask_root_dir):
    """
    gleason_type: 1, 2, 3, ...
    mask_dir: '  E:\A_bunch_of_data\Raw\Gleason_2019\raw_mask '

    """
    data_name_wos = (os.path.split(data_dir)[1]).split('.')[0]
    mask_name_ws = data_name_wos + '_nonconvex.jpg'
    doc = ['a', 'b', 'c', 'd', 'e', 'f']

    mask_num = 0
    for i in range(len(doc)):
        mask_first_dir = os.path.join(mask_root_dir, doc[i])
        mask_first_dir = os.path.join(mask_first_dir, gleason_type)
        mask_first_dir = os.path.join(mask_first_dir, mask_name_ws)
        if os.path.exists(mask_first_dir) :
            mask_num += 1
        else:
            continue

        if mask_num == 1:
            Y = np.array(Image.open(mask_first_dir)) == 255
        else:
            Y = Y + np.array(Image.open(mask_first_dir)) == 255






