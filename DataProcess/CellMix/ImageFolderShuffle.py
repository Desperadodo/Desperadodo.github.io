"""
'ImageFolderShuffle' ver 22 Nov 10
use 'shuffle_rate' to shuffle the img(mask) in a typical ImageFolder
the shuffle_rate means the ratio of images getting out of the OG folder
"""
import os
import random
import re
import csv
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm

name_dict = {}


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)


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
    return res


def save_file(f_image, from_dir, save_dir, suffix='.jpg', make_csv=False):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)
    if make_csv is True:
        name_dict[save_dir] = from_dir


def to_class(all_list, output_root, class_name, list_split, suffix, make_csv, mask=False):
    output_root = os.path.join(output_root, class_name)
    for i in list_split:
        if mask is True:
            img_dir = os.path.join(os.path.split(os.path.split(os.path.split(all_list[i])[0])[0])[0], 'mask')
            img_dir = os.path.join(img_dir, os.path.split(os.path.split(all_list[i])[0])[1])
            img_dir = os.path.join(img_dir, os.path.split(all_list[i])[1])
        else:
            img_dir = all_list[i]
        img = Image.open(img_dir)
        img_name = os.path.split(all_list[i])[1].split('.')[0]
        final_root = os.path.join(output_root, img_name)
        save_file(img, img_dir, final_root, suffix, make_csv)


def image_shuffle(input_root, output_root, shuffle_rate, is_mil=True, suffix='.jpg'):
    if is_mil is True:
        output_root = os.path.join(output_root, 'data')
        output_root_1 = os.path.join(os.path.split(output_root)[0], 'mask')
        input_root = os.path.join(input_root, 'data')
        input_root_1 = os.path.join(input_root, 'mask')
    class_names = os.listdir(input_root)
    print(class_names)
    class_num = len(class_names)
    shuffle_img = []

    for class_name in class_names:
        class_img_root = os.path.join(input_root, class_name)
        class_img_all = find_all_files(class_img_root, suffix=suffix)

        class_img_all_list = list(range(len(class_img_all)))
        print(class_img_all_list)
        random.shuffle(class_img_all_list)

        current_idx = 0
        shuffle_rate_flag = (len(class_img_all) * shuffle_rate) // (class_num - 1)


        for other_class_name in class_names:
            if other_class_name is not class_name:
                to_class(class_img_all,
                         output_root,
                         other_class_name,
                         class_img_all_list[int(current_idx):int(current_idx + shuffle_rate_flag)],
                         suffix,
                         make_csv=True)
                if is_mil is True:
                    to_class(class_img_all,
                             output_root_1,
                             other_class_name,
                             class_img_all_list[int(current_idx):int(current_idx + shuffle_rate_flag)],
                             suffix,
                             make_csv=False,
                             mask=True)
                current_idx = current_idx + shuffle_rate_flag

        to_class(class_img_all,
                 output_root,
                 class_name,
                 class_img_all_list[int(current_idx): -1],
                 suffix,
                 make_csv=True)
        if is_mil is True:
            to_class(class_img_all,
                     output_root_1,
                     class_name,
                     class_img_all_list[int(current_idx): -1],
                     suffix,
                     make_csv=False,
                     mask=True)


def main(input_root, output_root, shuffle_rate, is_mil=True, suffix='.jpg', split=True):
    if split is not True:
        image_shuffle(input_root, output_root, shuffle_rate, is_mil, suffix)
    else:
        image_folders = ['train', 'val', 'test']
        for image_folder in image_folders:
            current_input_root = os.path.join(input_root, image_folder)
            current_output_root = os.path.join(output_root, image_folder)

            image_shuffle(current_input_root, current_output_root, shuffle_rate, is_mil, suffix)


if __name__ == '__main__':
    input_root = r'E:\BaiduNetdiskDownload\WBC_MIL'
    output = r'D:\CellMix\WBC_MIL'
    shuffle_rate = 0.2
    for i in range(0, 4):
        output_root = output + '_' + str(shuffle_rate) + '_S'
        main(input_root, output_root,
             shuffle_rate=shuffle_rate, is_mil=True, suffix='.jpg',
             split=True)

        pd.DataFrame.from_dict(name_dict, orient='index', columns=['origin path']).to_csv(
            os.path.join(output_root, 'name_dict' + os.path.split(output_root)[1] + '.csv')
        )

        shuffle_rate = shuffle_rate + 0.2
