"""
Organize the data with .csv file to ensure that all data is in jpg format  ver： Aug 31th 19:46 official release
"""
import os
import re
import csv
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms
import notifyemail
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def del_file(filepath):
    # this is fucking dangerous
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
    del_file(file_pack_path)


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


def read_file(f_dir):
    """
    Read a file and convert it into numpy format
    """
    f_image = Image.open(f_dir)
    return f_image


def change_shape(image, corp_x=2400, corp_y=1800, f_x=1390, f_y=1038):
    """
    Corp then center place to maintain a same magnitude and Resize the image into x*y
    """
    if image.size[0] > corp_x or image.size[1] > corp_y:
        # Generate an object of CenterCrop class to crop the image from the center into corp_x*corp_y
        crop_obj = torchvision.transforms.CenterCrop((corp_y, corp_x))
        image = crop_obj(image)
        # print(image.size[0], image.size[1])

    image.thumbnail((f_x, f_y), Image.ANTIALIAS)
    return image


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    Save and rename the images, generate the renamed table
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)



def trans_csv_folder_to_imagefoder(target_path=r'C:\Users\admin\Desktop\MRAS_SEED_dataset',
                                   original_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_org_image',
                                   csv_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_label.csv'):
    """
    Original data format: a folder with image inside + a csv file with header which has the name and category of every image.
    Process original dataset and get data packet in image folder format
    :param target_path: the path of target image folder
    :param original_path: The folder with images
    :param csv_path: A csv file with header and the name and category of each image
    """
    idx = -1
    with open(csv_path, "rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        make_and_clear_path(target_path)  # Clear target_path
        for row in tqdm(rows):
            idx += 1
            if idx == 0:  # Skip the first header
                if not row[1].isdigit():
                    continue
            img_name = os.path.split(row[0])[1] if len(os.path.split(row[0])[1]) > 0 else row[0]
            item_path = os.path.join(original_path, img_name)
            if os.path.exists(os.path.join(target_path, row[1])):
                shutil.copy(item_path, os.path.join(target_path, row[1]))
            else:
                os.makedirs(os.path.join(target_path, row[1]))
                shutil.copy(item_path, os.path.join(target_path, row[1]))

        print('total num:', idx)


if __name__ == '__main__':
    trans_csv_folder_to_imagefoder(
        target_path=r'E:\Puzzle_Tuning_Datasets\Cropped\screened\Colonoscopy',
        original_path=r'E:\Puzzle_Tuning_Datasets\Cropped\unscreened\Colonoscopy',
        csv_path=r'C:\Users\8175M\Desktop\Pathology_Experiment\INF\inferance.csv'
    ) # ball ball可爱捏 哟 第五层
    # 蕾蕾最可爱 第三层的小雷雷
    #关注塔菲喵 关注永雏塔菲谢谢喵