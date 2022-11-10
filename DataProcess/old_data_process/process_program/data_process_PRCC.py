import os
import re
import csv
import shutil
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
# from scipy.io import loadmat
import torchvision.transforms
'''
                  _oo0oo_
                 o8888888o
                 88" . "88
                 (| -_- |)
                  0\ = /0
               ___/`---'\___
               .' \\| |// '.
             / \\||| : |||// \
           / _||||| -:- |||||- \
             | | \\\ - /// | |
           | \_| ''\---/'' |_/ |
            \ .-\__ '-' ___/-. /
         ___'. .' /--.--\ `. .'___
      ."" '< `.___\_<|>_/___.' >' "".
     | | : `- \`.;`\ _ /`;.`/ - ` : | |
       \ \ `_. \_ __\ /__ _/ .-` / /
=====`-.____`.___ \_____/___.-`___.-'=====
`=---='
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
佛祖保佑 永不宕机 永⽆BUG
'''

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


"""def MatrixToImage(data):
    data = (data != 0)*255
    new_im = data.astype(np.uint8)
    return new_im"""


"""def read_reform_mat(mat_dir, save_dir, key='Type',  suffix='.jpg'):
    # save_dir is with the name but without the suffix
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    m = loadmat(mat_dir)
    daaa = m.keys()
    data = m[key]
    new_im = MatrixToImage(data)
    cv2.imwrite(save_dir+suffix, new_im)"""


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    cv2.imwrite(save_dir + suffix, f_image)


def pc_to_stander(root_from_data, root_from_mask, root_to):
    """root_target = root_to
    make_and_clear_path(root_target)"""

    data_dir_list = find_all_files(root=root_from_data, suffix=".png")
    # mask_dir_list = find_all_files(root=root_from_mask, suffix=".mat")
    print(len(data_dir_list))
    name_dict = {}

    """for data_seq in tqdm(range(len(data_dir_list))):
        f_dir = data_dir_list[data_seq]
        f_img = cv2.imread(f_dir)
        img_name = os.path.split(f_dir)[1].split('.')[0]

        data_save_dir = os.path.join(os.path.join(root_target, 'data'), '2')
        data_save_dir = os.path.join(data_save_dir, img_name)
        mask_save_dir = os.path.join(os.path.join(root_target, 'mask'), '2')
        mask_save_dir = os.path.join(mask_save_dir, img_name)

        name_dict[data_save_dir] = f_dir

        for mask_seq in range(len(mask_dir_list)):
            mask_name = os.path.split(mask_dir_list[mask_seq])[1].split('.')[0]
            if mask_name in img_name:
                read_reform_mat(mask_dir_list[mask_seq], mask_save_dir)
                break
            else:
                continue

        save_file(f_img, data_save_dir)

    root_target, _ = os.path.split(root_to)
    root_target, _ = os.path.split(root_target)
    pd.DataFrame.from_dict(name_dict, orient='index', columns=['origin path']).to_csv(
        os.path.join(root_target, 'name_dict_PCam_test.csv')
    )"""


if __name__ == '__main__':
    pc_to_stander(root_from_data=r'/Users/munros/Downloads/1',
                  root_from_mask=r'E:\A_bunch_of_data\Raw\pRCC\nuclei_prediction',
                  root_to=r"E:\A_bunch_of_data\Processed\pRCC")

    # find_all_files(r'E:\A_bunch_of_data\Raw\gastric_image\train_org_image', 'png')












