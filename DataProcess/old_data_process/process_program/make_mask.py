import os
import re
import csv
import shutil

import PIL.Image
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    cv2.imwrite(save_dir + suffix, f_image)
    # f_image.save(save_dir + suffix)


def make_mask(root_from, root_to):
    make_and_clear_path(root_to)
    f_dir_list = find_all_files(root=root_from, suffix=".png")

    # print(f_dir_list)
    name_dict = {}
    i = 0
    trigger = [1, 2, 3, 4, 5, 6]
    for x in range(len(trigger)):
        for seq in tqdm(range(len(f_dir_list))):
            f_dir = f_dir_list[seq]

            # print(len(Image.open(f_dir).split()))
            f_img = np.array(Image.open(f_dir))

            # print(type(trigger[0]))
            img_name = (os.path.split(f_dir)[1]).split('.')[0]
            Y = f_img == trigger[x]
            f_img = Y * 255
            save_path = os.path.join(root_to, str(trigger[x]))
            save_path = os.path.join(save_path, img_name)
            save_file(f_img, save_path)

        """f_img = np.all(f_img == trigger) * 255
        processed_img = PIL.Image.fromarray(np.uint8(f_img))
        processed_img.save(f_dir)"""




if __name__ == '__main__':
    make_mask(root_from=r'E:\A_bunch_of_data\Raw\Gleason_2019\Maps1_T', root_to=r"E:\A_bunch_of_data\Raw\Gleason_2019\Maps1")