import os
import re
import csv
import shutil
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
    print(res)
    print(len(res))
    return res


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)


def pc_to_stander(root_from, root_to):
    root_target = root_to
    make_and_clear_path(root_target)

    f_dir_list = find_all_files(root=root_from, suffix=".png")
    print(f_dir_list)
    name_dict = {}
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0

    for seq in tqdm(range(len(f_dir_list))):
        f_dir = f_dir_list[seq]
        _, str = os.path.split(f_dir)
        mp = str.split("-")[-2]
        type = (str.split("_")[2]).split("-")[0]
        name = str.split(".")[0]
        print(mp)
        print(type)

        f_img = Image.open(f_dir)
        if mp == '40':
            root_target = os.path.join(root_to, '40')
        elif mp == '100':
            root_target = os.path.join(root_to, '100')
        elif mp == '200':
            root_target = os.path.join(root_to, '200')
        else:
            root_target = os.path.join(root_to, '400')
        if type == 'DC':
            root_target = os.path.join(root_target, 'ductal_carcinoma')
        elif type == 'LC':
            root_target = os.path.join(root_target, 'lobular_carcinoma')
        elif type == 'MC':
            root_target = os.path.join(root_target, 'mucinous_carcinoma')
        elif type == 'PC':
            root_target = os.path.join(root_target, 'papillary_carcinoma')
        elif type == 'A':
            root_target = os.path.join(root_target, 'adenosis')
        elif type == 'F':
            root_target = os.path.join(root_target, 'fibroadenoma')
        elif type == 'PT':
            root_target = os.path.join(root_target, 'phyllodes_tumor')
        else:
            root_target = os.path.join(root_target, 'tubular_adenoma')

        save_dir = os.path.join(root_target, name)


        name_dict[save_dir] = f_dir

        save_file(f_img, save_dir)

    root_target, _ = os.path.split(root_to)
    root_target, _ = os.path.split(root_target)
    pd.DataFrame.from_dict(name_dict, orient='index', columns=['origin path']).to_csv(
        os.path.join(root_target, 'name_dict_BreaKHis.csv')
    )


if __name__ == '__main__':
    pc_to_stander(root_from=r'E:\A_bunch_of_data\Raw\BreaKHis_v1\histology_slides\breast',
                  root_to=r'E:\A_bunch_of_data\Processed\BreaKHis_jpg')
    """find_all_files(root=r'E:\A_bunch_of_data\Raw\BreaKHis_v1\histology_slides\breast'
                   , suffix='.png')"""









