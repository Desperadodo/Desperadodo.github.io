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
    return res


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)


"""def change_shape(image, f_x=1390, f_y=1038):先不要这个，图片分割剪切在模型里面
    '''
    将图片大小改为x*y
    '''
    corp_ratio = float(max(f_y, f_x)) / float(min(f_y, f_x))

    minarri, posshort = min(image.size[0], image.size[1])  # posshort=0 or 1
    posslong = 1-posshort  # qu fan
    
    maxarri = int(corp_ratio * minarri)
    
    rotate_obj= torchvision.transforms.RandomRotation(90)
    if posshort ==1:
        image = rotate_obj(image)

    # 生成一个CenterCrop类的对象,用来将图片从中心裁剪成corp_x  corp_y
    crop_obj = torchvision.transforms.RandomCrop((minarri,maxarri))
    image = crop_obj(image)

    crop_obj = torchvision.transforms.RandomCrop((minarri, maxarri))
    # print(image.size[0], image.size[1])

    image.thumbnail((f_x, f_y), Image.ANTIALIAS)
    return image"""


def pc_to_stander(root_from, root_to):
    root_target = root_to
    make_and_clear_path(root_target)

    f_dir_list = find_all_files(root=root_from, suffix=".bmp")
    print(f_dir_list)
    name_dict = {}
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0

    for seq in tqdm(range(len(f_dir_list))):
        f_dir = f_dir_list[seq]

        f_img = Image.open(f_dir)


        if 'im_Superficial-Intermediate' in f_dir:
            if 'CROPPED' in f_dir:
                root_target = os.path.join(root_to, 'Cropped')
            else:
                root_target = os.path.join(root_to, "Full")
            root_target = os.path.join(root_target, 'Sup_Intermediate')
            i1 += 1
            save_dir = os.path.join(root_target, str(i1))
        elif 'im_Parabasal' in f_dir:
            if 'CROPPED' in f_dir:
                root_target = os.path.join(root_to, 'Cropped')
            else:
                root_target = os.path.join(root_to, "Full")
            root_target = os.path.join(root_target, 'Parabasal')
            i2 += 1
            save_dir = os.path.join(root_target, str(i2))
        elif 'im_Dyskeratotic' in f_dir:
            if 'CROPPED' in f_dir:
                root_target = os.path.join(root_to, 'Cropped')
            else:
                root_target = os.path.join(root_to, "Full")
            root_target = os.path.join(root_target, 'Dyskeratotic')
            i3 += 1
            save_dir = os.path.join(root_target, str(i3))
        elif 'im_Koilocytotic' in f_dir:
            if 'CROPPED' in f_dir:
                root_target = os.path.join(root_to, 'Cropped')
            else:
                root_target = os.path.join(root_to, "Full")
            root_target = os.path.join(root_target, 'Koilocytotic')
            i4 += 1
            save_dir = os.path.join(root_target, str(i4))
        else:
            if 'CROPPED' in f_dir:
                root_target = os.path.join(root_to, 'Cropped')
            else:
                root_target = os.path.join(root_to, "Full")
            root_target = os.path.join(root_target, 'Metaplastic')
            i5 += 1
            save_dir = os.path.join(root_target, str(i5))

        name_dict[save_dir] = f_dir

        save_file(f_img, save_dir)

    pd.DataFrame.from_dict(name_dict, orient='index', columns=['origin path']).to_csv(
        os.path.join(r"E:\A_bunch_of_data", 'name_dict_cervical.csv')
    )


if __name__ == '__main__':
    pc_to_stander(root_from=r'E:\A_bunch_of_data\Raw\cervical_image',
                  root_to=r'E:\A_bunch_of_data\Processed\cervical_jpg')
