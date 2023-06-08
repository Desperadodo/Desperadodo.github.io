import os

os.add_dll_directory(r"D:\chrome_download\github220901\openslide-win64\bin")
# 注意openslide的使用需要这样 另外叫将openslide添加到PATh里面
import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import io
from PIL import Image, ImageTk
import json
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading
import multiprocessing

import shutil
import PIL.Image as Image
import numpy as np
import torch
from tqdm import tqdm
import cv2
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

IMAGE_SIZE = (2000, 2000)

"""result_path = r'H:\PuzzleTuning\SNL-Breast'
slide = openslide.open_slide(
    r'E:\Puzzle_Tuning_Datasets\Raw\Breast-Metastases-MSKCC(SNL-Breast)\HobI16-053768896760.svs')
highth = 2000
width = highth
data_gen = DeepZoomGenerator(slide, tile_size=highth, overlap=0, limit_bounds=False)
print(data_gen.tile_count)
print(data_gen.level_count)

[w, h] = slide.dimensions

num_w = int(np.floor(w / width)) + 1
num_h = int(np.floor(h / highth)) + 1
for i in range(num_w):
    for j in range(num_h):
        img = np.array(data_gen.get_tile(15, (i, j)))  # 切图
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img.save(os.path.join(result_path, "HobI16-053768896760" + str(i) + '_' + str(j) + ".png"))  # 保存图像"""


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


def convert_to_npy(a_data_path, patch_size):
    patch_size = to_2tuple(patch_size)

    # 处理转换

    # 传回npy
    img = Image.open(a_data_path)
    w, h = img.size
    factor = min(w // patch_size[0], h // patch_size[1])
    numpy_img = img.crop([0, 0, factor * patch_size[0], factor * patch_size[1]])
    numpy_img = np.array(numpy_img)

    return numpy_img


class to_patch:
    """
    Split a image into patches, each patch with the size of patch_size
    """

    def __init__(self, patch_size=(16, 16)):
        patch_size = to_2tuple(patch_size)
        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]

    def __call__(self, x):
        x = torch.tensor(x)
        x = x.permute(2, 0, 1)
        c, h, w = x.shape
        # print(x.shape)
        # assert h // self.patch_h == h / self.patch_h and w // self.patch_w == w / self.patch_w

        num_patches = (h // self.patch_h) * (w // self.patch_w)

        # patch encoding
        # (c, h, w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, self.patch_h, self.patch_w, c)
        # -> (n_patches, patch_size^2*c)
        patches = x.view(
            c,
            h // self.patch_h,
            self.patch_h,
            w // self.patch_w,
            self.patch_w).permute(1, 3, 2, 4, 0).reshape(num_patches, -1)  # it can also used in transformer Encoding

        # patch split
        # (n_patches, patch_size^2*c)
        # -> (num_patches, self.patch_h, self.patch_w, c)
        # -> (num_patches, c, self.patch_h, self.patch_w)
        patches = patches.view(num_patches,
                               self.patch_h,
                               self.patch_w,
                               c).permute(0, 3, 1, 2)
        return patches


def to_2tuple(input):
    if type(input) is tuple:
        if len(input) == 2:
            return input
        else:
            if len(input) > 2:
                output = (input[0], input[1])
                return output
            elif len(input) == 1:
                output = (input[0], input[0])
                return output
            else:
                print('cannot handle none tuple')
    else:
        if type(input) is list:
            if len(input) == 2:
                output = (input[0], input[1])
                return output
            else:
                if len(input) > 2:
                    output = (input[0], input[1])
                    return output
                elif len(input) == 1:
                    output = (input[0], input[0])
                    return output
                else:
                    print('cannot handle none list')
        elif type(input) is int:
            output = (input, input)
            return output
        else:
            print('cannot handle ', type(input))
            raise ('cannot handle ', type(input))


k = 0


def cut_to_patch(slide, save_root, patch_size, img_name):
    w, h = slide.level_dimensions[0]
    global k
    for i in range(4, w // IMAGE_SIZE[0] - 5):
        for j in range(4, h // IMAGE_SIZE[1] - 5):
            patch = slide.read_region((i * 2000, j * 2000), 0, (2000, 2000))
            patch = patch.convert('RGB')
            # print('finish id:%d image' % image_list.index(id))
            # patch = patch.resize((2000, 2000),Image.ANTIALIAS)
            # 统一归为384*384
            # save_file(patch, os.path.join(save_root_0, img_name + '-' + str((i + 1) * (j + 1))))
            img_single = patch.resize((1, 1), Image.ANTIALIAS)
            r, g, b = img_single.getpixel((0, 0))
            if r < 220 and g < 190 and b < 190 and r > 100 and g > 50 and b > 50 and k < 3000 and r > g + 30:
                save_file(patch, os.path.join(save_root, img_name + '-' + str(i) + '-' + str(j)))
                k = k + 1

            else:

                continue
                # save_file(patch, os.path.join('H:\PuzzleTuning\SNL-Breast-Back', img_name + '-' + str(i) + '-' + str(j)))


def read_and_convert(data_root, save_root_0, suffix=None, patch_size=(96, 96)):
    # 一次处理只一个数据集, 每个数据集的处理方式可能有不同

    # 读入所有数据
    # class_names = os.listdir(data_root)
    class_names = ['']
    for class_name in class_names:
        global k
        k = 0
        class_root = os.path.join(data_root, class_name)
        save_root = os.path.join(save_root_0, class_name)
        all_files = find_all_files(class_root, suffix)
        make_and_clear_path(save_root)

        # 把所有数据转换为同一个格式
        for img in all_files:
            slide = openslide.open_slide(img)
            img_name = os.path.split(img)[1].split('.')[0]
            cut_to_patch(slide, save_root, patch_size, img_name)


if __name__ == '__main__':
    read_and_convert(r'D:\Puzzle Tuning Datasets\raw',
                     r'H:\PuzzleTuning\2000',
                     'svs')
