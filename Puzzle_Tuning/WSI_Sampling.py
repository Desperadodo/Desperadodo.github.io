"""
This code is mainly used
"""
import os
import random
import shutil
import PIL.Image as Image
import tqdm
import numpy as np
import torch
from tqdm import tqdm
import cv2
from torchvision import transforms
from PIL import ImageFile
import pandas as pd


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)
    # cv2.imwrite(save_dir+suffix, f_image)


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    if type(suffix) is tuple or type(suffix) is list:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None:
                    status = 0
                    for i in suffix:
                        if not f.endswith(i):
                            pass
                        else:
                            status = 1
                            break
                    if status == 0:
                        continue
                res.append(os.path.join(root, f))
        return res

    elif type(suffix) is str or suffix is None:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None and not f.endswith(suffix):
                    continue
                res.append(os.path.join(root, f))
        return res

    else:
        print('type of suffix is not legal :', type(suffix))
        return -1


def rand_file(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    picknumber1 = 1000  # 按照rate比例从文件夹中取一定数量的文件
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    return sample1


def img_sample(img_name, goal_length):
    # 小图片在大图片上进行一个几何上均匀的采样，3840 960 384 96的比例为1:8:8:8
    output = []
    img_real_name = img_name.split('-')[:-3]
    img_real_name = "-".join(img_real_name)  # 后三位分别是图片边长，图片在原图的x，在原图的y
    img_length = int(img_name.split('-')[-3])
    og_x = int(img_name.split('-')[-2])
    og_y = int(img_name.split('-')[-1])

    if goal_length == 960:  # 960的情况将对应3840来, 与3840为1：8的关系
        for i in range(4):
            for j in range(4):
                goal_x = int(og_x * (img_length / goal_length) + i)
                goal_y = int(og_y * (img_length / goal_length) + j)
                goal_img_name = img_real_name + '-' + str(goal_length) + '-' + str(goal_x) + '-' + str(goal_y)
                output.append(goal_img_name)

    elif goal_length == 384:  # 384的情况对应3840来，与3840为1：8的关系
        for i in range(2):
            for j in range(2):
                samples = [(0, 0), (4, 0), (1, 1), (3, 1), (1, 3), (3, 3), (0, 4), (4, 4)]
                for sample in samples:
                    goal_x = int(og_x * (img_length / goal_length) + sample[0] + i*5)
                    goal_y = int(og_y * (img_length / goal_length) + sample[1] + j*5)
                    goal_img_name = img_real_name + '-' + str(goal_length) + '-' + str(goal_x) + '-' + str(goal_y)
                    output.append(goal_img_name)
    elif goal_length == 96:  #
        goal_x = int(og_x * (img_length / goal_length) + 1)
        goal_y = int(og_y * (img_length / goal_length) + 2)
        goal_img_name = img_real_name + '-' + str(goal_length) + '-' + str(goal_x) + '-' + str(goal_y)
        output.append(goal_img_name)


    return output


def Sample(XL_pth, L_pth, M_pth, S_pth, output_pth, ds_name):
    XL_out_pth = os.path.join(output_pth + '\\L', ds_name + '_3840')
    L_out_pth = os.path.join(output_pth + '\\L', ds_name + '_960')
    M_out_pth = os.path.join(output_pth + '\\M', ds_name + '_384')
    S_out_pth = os.path.join(output_pth + '\\S', ds_name + '_96')

    for pth in [XL_out_pth, L_out_pth, M_out_pth, S_out_pth]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    XL_imgs_s = os.listdir(XL_pth)
    XL_num = len(XL_imgs_s)
    if XL_num > 1000:
        XL_imgs_s = rand_file(XL_pth)

    for XL_img_seq in tqdm(range(len(XL_imgs_s))):  # XL_img_pth is a absolute path, which looks like /xxx/3840/xxxx-3840/abcd-3840-2-3.jpg
        XL_img_s = XL_imgs_s[XL_img_seq]
        XL_img = XL_img_s.split('.')[0]

        # XL(3840) resizing & saving
        img = Image.open(os.path.join(XL_pth, XL_img_s))
        resized_img = img.resize((384, 384), Image.ANTIALIAS)
        save_file(resized_img, os.path.join(XL_out_pth, XL_img))

        L_imgs = img_sample(XL_img, 960)
        for L_img in L_imgs:  # L_img is just a name, which looks like abcd-960-20-30
            L_img_pth = os.path.join(L_pth, L_img + '.jpg')
            if os.path.exists(L_img_pth):
                shutil.copy(L_img_pth, os.path.join(L_out_pth, L_img + '.jpg'))

        M_imgs = img_sample(XL_img, 384)
        for M_img in M_imgs:  # L_img is just a name, which looks like abcd-960-20-30
            M_img_pth = os.path.join(M_pth, M_img + '.jpg')
            if os.path.exists(M_img_pth):
                shutil.copy(M_img_pth, os.path.join(M_out_pth, M_img + '.jpg'))

                S_imgs = img_sample(M_img, 96)
                for S_img in S_imgs:
                    S_img_pth = os.path.join(S_pth, S_img + '.jpg')
                    if os.path.exists(S_img_pth):
                        shutil.copy(S_img_pth, os.path.join(S_out_pth, S_img + '.jpg'))


def main(datasets_root_1, datasets_root_2, datasets_out_root):

    dataset_csv = find_all_files(r'I:\csv_I', '.csv')
    dataset_csv = ['I:\PuzzleTuning3.0\96\PAIP2020-96.csv', 'I:\PuzzleTuning3.0\96\PAIP2021-96.csv']
    print(dataset_csv)
    for dataset in dataset_csv:
        dataset = os.path.split(dataset)[1]
        dataset_name = dataset.split('-')[:-1]
        dataset_name = "-".join(dataset_name)
        print(dataset_name)
        XL = os.path.join(datasets_root_2 + '\\3840', dataset_name + '-3840')
        L = os.path.join(datasets_root_2 + '\\960', dataset_name + '-960')
        M = os.path.join(datasets_root_2 + '\\384', dataset_name + '-384')
        S = os.path.join(datasets_root_2 + '\\96', dataset_name + '-96')

        Sample(XL, L, M, S, datasets_out_root, dataset_name)


if __name__ == '__main__':
    main(datasets_root_1=r'H:\PuzzleTuning3.0',
         datasets_root_2=r'I:\PuzzleTuning3.0',
         datasets_out_root=r'D:\MJ\CPIA_MJ')



