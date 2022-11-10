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


def pc_to_stander(root_from, root_to):
    root_target = root_to
    make_and_clear_path(root_target)

    f_dir_list = find_all_files(root=root_from, suffix=".bmp")
    print(f_dir_list)
    name_dict = {}

    with open(r'E:\A_bunch_of_data\Raw\Warwick QU Dataset (Released 2016_07_08)\Grade.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        length = len(result)

    for seq in tqdm(range(len(f_dir_list))):
        f_dir = f_dir_list[seq]
        f_img = Image.open(f_dir)
        _, img_name = os.path.split(f_dir)
        img_name = img_name.split('.')[0]

        if img_name[0:4] == 'test':
            save_dir = os.path.join(root_target, 'test')
        else:
            save_dir = os.path.join(root_target, 'train')

        if img_name[-4:] == 'anno':
            save_dir = os.path.join(save_dir, 'mask')
            img_name = img_name[0:-5]
        else:
            save_dir = os.path.join(save_dir, 'data')

        for i in range(length):
            # print(img_name.split('.')[0])
            l = len(result[i][0])
            if img_name == result[i][0]:
                if result[i][2] == ' malignant':
                    save_dir = os.path.join(save_dir, 'malignant')
                    print('1')
                else:
                    save_dir = os.path.join(save_dir, 'benign')
                break
            else:
                continue
        save_dir = os.path.join(save_dir, img_name.split('.')[0])
        name_dict[save_dir] = f_dir
        save_file(f_img, save_dir)

    root_target, _ = os.path.split(root_to)
    root_target, _ = os.path.split(root_target)
    pd.DataFrame.from_dict(name_dict, orient='index', columns=['origin path']).to_csv(
        os.path.join(root_target, 'name_dict_warwick.csv')
    )


if __name__ == '__main__':
    pc_to_stander(root_from=r'E:\A_bunch_of_data\Raw\Warwick QU Dataset (Released 2016_07_08)',
                  root_to=r"E:\A_bunch_of_data\Processed\Warwick")

    """find_all_files(r'E:\A_bunch_of_data\Raw\gastric_image\train_org_image', 'png')"""
