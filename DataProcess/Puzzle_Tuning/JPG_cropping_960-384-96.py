"""
'JPG_cropping_960...'  ver： 22 Nov 10
Crop pathology images into patches  Using average filtering to screen the useful pieces which are mostly red/purple

Specially mod ver
maximize the efficient of cropping in different size
"""

import os
import shutil
import PIL.Image as Image
import numpy as np
import torch
from tqdm import tqdm
import cv2
from torchvision import transforms
from PIL import ImageFile
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


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


def convert_to_npy(a_data_path, patch_size=(960, 960)):
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

        h_1 = (h // self.patch_h) * self.patch_h
        w_1 = (w // self.patch_w) * self.patch_w
        x = x[:, ((h-h_1)//2):((h-h_1)//2+h_1), ((w-w_1)//2):((w-w_1)//2+w_1)]
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

        '''
        # check
        for i in range(len(patches)):
            recons_img = ToPILImage()(patches[i])
            recons_img.save(os.path.join('./patch_play', 'recons_target'+str(i)+'.jpg'))


        # patch compose to image
        # (num_patches, c, self.patch_h, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, c, self.patch_h, self.patch_w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (c, h, w)
        patches = patches.view(h // self.patch_h,
                               w // self.patch_w,
                               c,
                               self.patch_h,
                               self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        '''

        '''
        # visual check
        # reshape
        composed_patches = patches.view(h // self.patch_h,
                                        w // self.patch_w,
                                        c,
                                        self.patch_h,
                                        self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        # view pic
        from torchvision.transforms import ToPILImage
        composed_img = ToPILImage()(bag_image[0])  # transform tensor image to PIL image
        composed_img.save(os.path.join('./', 'composed_img.jpg'))

        '''

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


def pick_patch(patch):
    img_single = patch.resize((1, 1), Image.ANTIALIAS)
    r, g, b = img_single.getpixel((0, 0))
    if r > 200 and g > 180 and b > 180:
        return False
    else:
        return True


def cut_to_patch(img, save_root, patch_size_0, patch_size_1, patch_size_2, img_name, class_name,
                 name_dir_0, name_dir_1, name_dir_2):
    numpy_img = convert_to_npy(img)
    patch_size_num_0 = patch_size_0[0]
    patch_size_num_1 = patch_size_1[0]
    patch_size_num_2 = patch_size_2[0]
    save_root_0 = os.path.join(os.path.join(save_root, str(patch_size_num_0)), class_name + '-' + str(patch_size_num_0))
    save_root_1 = os.path.join(os.path.join(save_root, str(patch_size_num_1)), class_name + '-' + str(patch_size_num_1))
    save_root_2 = os.path.join(os.path.join(save_root, str(patch_size_num_2)), class_name + '-' + str(patch_size_num_2))



    img_split_0 = to_patch(patch_size_0)
    img_patches_0 = img_split_0(numpy_img)
    i = 0
    j = 0
    k = 0
    for patch in img_patches_0:
        patch = patch.permute(1, 2, 0)
        patch = patch.numpy()
        if pick_patch(Image.fromarray(patch.astype('uint8')).convert('RGB')):
            img_split_1 = to_patch(patch_size_1)
            img_patches_1 = img_split_1(patch)
            for patch_1 in img_patches_1:
                patch_1 = patch_1.permute(1, 2, 0)
                patch_1 = patch_1.numpy()
                if pick_patch(Image.fromarray(patch_1.astype('uint8')).convert('RGB')):
                    img_split_2 = to_patch(patch_size_2)
                    img_patches_2 = img_split_2(patch_1)
                    for patch_2 in img_patches_2:
                        patch_2 = patch_2.permute(1, 2, 0)
                        patch_2 = patch_2.numpy()
                        if pick_patch(Image.fromarray(patch_2.astype('uint8')).convert('RGB')):
                            k = k + 1
                            if k % 10 == 0:
                                patch_2 = Image.fromarray(patch_2.astype('uint8')).convert('RGB')
                                save_dir_2 = os.path.join(save_root_2,
                                                          img_name + '-' + str(patch_size_num_2) + '-' + str(k//10))
                                name_dir_2[save_dir_2] = img
                                save_file(patch_2, save_dir_2)
                    patch_1 = Image.fromarray(patch_1.astype('uint8')).convert('RGB')
                    save_dir_1 = os.path.join(save_root_1, img_name + '-' + str(patch_size_num_1) + '-' + str(j))
                    name_dir_1[save_dir_1] = img
                    save_file(patch_1, save_dir_1)
                    j = j + 1
            patch = Image.fromarray(patch.astype('uint8')).convert('RGB')
            patch = patch.resize((384, 384), Image.ANTIALIAS)  # 归为384*384
            save_dir_0 = os.path.join(save_root_0, img_name + '-' + str(patch_size_num_0) + '-' + str(i))
            name_dir_0[save_dir_0] = img
            # 保存相关.csv

            save_file(patch, save_dir_0)
            i = i + 1
    pd.DataFrame.from_dict(name_dir_0, orient='index', columns=['origin path']).to_csv(
        os.path.join(os.path.join(save_root,
                                  str(patch_size_num_0)), class_name + '-' + str(patch_size_num_0) + '.csv')
    )
    pd.DataFrame.from_dict(name_dir_1, orient='index', columns=['origin path']).to_csv(
        os.path.join(os.path.join(save_root,
                                  str(patch_size_num_1)), class_name + '-' + str(patch_size_num_1) + '.csv')
    )
    pd.DataFrame.from_dict(name_dir_2, orient='index', columns=['origin path']).to_csv(
        os.path.join(os.path.join(save_root,
                                  str(patch_size_num_2)), class_name + '-' + str(patch_size_num_2) + '.csv')
    )

def read_and_convert(data_root, save_root, suffix=None, patch_size=None):
    # 一次处理只一个数据集, 每个数据集的处理方式可能有不同

    # 读入所有数据
    if patch_size is None:
        patch_size = [(960, 960), (384, 384), (96, 96)]

    # class_names = os.listdir(data_root)
    class_names = ['CRC_v1', 'GEC_v1', 'LCA_v1', 'MEL_v1']

    for class_name in class_names:
        class_root = os.path.join(data_root, class_name)
        all_files = find_all_files(class_root, suffix)
        make_and_clear_path(save_root)
        name_dir_0 = {}
        name_dir_1 = {}
        name_dir_2 = {}
        # 把所有数据转换为同一个格式
        for img in all_files:
            img_name = os.path.split(img)[1].split('.')[0]
            cut_to_patch(img, save_root, patch_size[0], patch_size[1], patch_size[2], img_name, class_name,
                         name_dir_0, name_dir_1, name_dir_2)


if __name__ == '__main__':
    read_and_convert(r'H:\PuzzleTuning\2000',
                     r'H:\PuzzleTuning',
                     'jpg')


