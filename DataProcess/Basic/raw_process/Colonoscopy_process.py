"""
数据集Colonoscopy：1.0: 按照最大内接正方形的做法进行裁剪，暂时不统一resize

"""
from tkinter import *
import os
import re
import csv
import shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from tqdm import tqdm
import torchvision.transforms


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    在这个任务中，需要只寻找名称中带有'mask'的图片
    """
    res = []
    for root, _, files in os.walk(root):
        print(files)
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            f1 = f.split('.')[0]
            if f1.endswith('mask') == 0:
                continue
            res.append(os.path.join(root, f))
    return res


def cut_resize(f_image, resize=False):
    """
    :param f_image:
    :param resize: image.shape 的格式
    :return:
    """
    sp = f_image.size
    width = sp[0]  # height(rows) of image
    height = sp[1]  # width(colums) of image
    if width >= height:
        box = ((width - height) / 2, 0, (width + height) / 2, height)
        cropped = f_image.crop(box)

        # shorter = cols
        # cropped = f_image[int((rows - cols) / 2):int((rows - cols) / 2 + shorter), 0:shorter]  # 裁剪坐标为[y0:y1, x0:x1]
    else:
        box = (0, (height - width) / 2, width, (height + width) / 2)
        cropped = f_image.crop(box)
        # shorter = rows
        # cropped = f_image[0:shorter, int((cols - rows) / 2):int((cols - rows) / 2 + shorter)]  # 裁剪坐标为[y0:y1, x0:x1]

    return cropped


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)


def main_process_positive(root_from, root_to):
    root_target = root_to
    root_target_mask = os.path.join(os.path.join(root_target, 'mask'), 'Positive')
    root_target_data = os.path.join(os.path.join(root_target, 'data'), 'Positive')
    make_and_clear_path(root_target_data)
    make_and_clear_path(root_target_mask)
    suffix = '.jpg'

    f_dir_list = find_all_files(root=root_from, suffix=".jpg")

    for seq in tqdm(range(len(f_dir_list))):
        mask_dir = f_dir_list[seq]
        mask_img = Image.open(mask_dir)

        img_name = os.path.split(mask_dir)[1].split('.')[0][: -5]
        img_name_s = img_name + suffix

        data_dir = os.path.join(os.path.split(mask_dir)[0], img_name_s)
        data_img = Image.open(data_dir)

        mask_img_cropped = cut_resize(mask_img)
        mask_save_dir = os.path.join(root_target_mask, img_name)
        save_file(mask_img_cropped, mask_save_dir)

        data_img_cropped = cut_resize(data_img)
        data_save_dir = os.path.join(root_target_data, img_name)
        save_file(data_img_cropped, data_save_dir)


# def main_process_negative(root_from, root_to):




if __name__ == '__main__':
    main_process_positive(
        root_from=r'/Users/munros/MIL/Raw/Colonoscopy/tissue-train-pos-v1',
        root_to=r'/Users/munros/MIL/Processed/Colonoscopy'
    )
