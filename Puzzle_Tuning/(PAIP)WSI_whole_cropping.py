import os

os.add_dll_directory(r"D:\chrome_download\github220901\openslide-win64\bin")
# 注意openslide的使用需要这样 另外叫将openslide添加到PATh里面
import shutil
import PIL.Image as Image
import numpy as np
import openslide
import torch
from tqdm import tqdm
import cv2
from torchvision import transforms
from PIL import ImageFile
import pandas as pd
import xml.etree.ElementTree as ET

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

STANDARD_MPP = 0.4942
patch_size = [(3840, 3840), (960, 960), (384, 384), (96, 96)]


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
        x = x[:, ((h - h_1) // 2):((h - h_1) // 2 + h_1), ((w - w_1) // 2):((w - w_1) // 2 + w_1)]
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


def xml2mask_pos(img_name, data_root, save_root, type, patch_size = 2560):
    img = os.path.join(data_root, img_name + '.svs')
    print(img)
    xml = os.path.join(data_root, img_name + '.xml')

    slide = openslide.open_slide(img)
    w, h = slide.level_dimensions[0]
    mask = np.zeros([h, w], dtype=np.uint8)

    tree = ET.parse(xml)
    root = tree.getroot()
    Annotations = root.findall('Annotation')
    for Annotation in Annotations:
        id = Annotation.attrib['Id']
        if id == "2": # id = 2 rep for viable tumor region
            regions = Annotation.findall('Regions/Region')
            for region in regions:
                NegativeROA = region.attrib['NegativeROA']
                if NegativeROA == "0": # NROA = 0 rep for none background
                    points = []
                    for point in region.findall('Vertices/Vertex'):
                        x = float(point.attrib['X'])
                        y = float(point.attrib['Y'])
                        points.append([x, y])
                    pts = np.asarray([points], dtype=np.int32)
                    cv2.fillPoly(img=mask, pts=pts, color=255)

    for i in range(1, w // patch_size - 1):
        for j in range(1, h // patch_size - 1):
            mask_patch = mask[j*patch_size:(j+1)*patch_size , i*patch_size:(i+1)*patch_size]
            if mask_patch.mean(axis=0).mean(axis=0) > 102 and mask_patch.mean(axis=0).mean(axis=0) < 200:  # 40% * 255
                # save mask
                mask_patch = cv2.resize(mask_patch, (384, 384))
                cv2.imwrite(save_root + "\\mask\\" + str(type) + '\\' + img_name + '-' + str(i) + '-' + str(j)+'.jpg', mask_patch)
                img_patch = slide.read_region(
                    (i*patch_size, j*patch_size),
                    0,
                    (patch_size, patch_size)
                )
                patch = img_patch.convert('RGB')
                # save data
                patch = patch.resize((384, 384), Image.ANTIALIAS)
                save_file(patch, save_root + "\\data\\" + str(type) + '\\' + img_name + '-' + str(i) + '-' + str(j))
            else:
                continue

def xml2mask_neg(img_name, data_root, save_root, type, patch_size = 2560):
    img = os.path.join(data_root, img_name + '.svs')
    print(img)
    xml = os.path.join(data_root, img_name + '.xml')

    slide = openslide.open_slide(img)
    w, h = slide.level_dimensions[0]
    mask = np.zeros([h, w], dtype=np.uint8)

    tree = ET.parse(xml)
    root = tree.getroot()
    Annotations = root.findall('Annotation')
    for Annotation in Annotations:
        id = Annotation.attrib['Id']
        if id == "2": # id = 2 rep for viable tumor region
            regions = Annotation.findall('Regions/Region')
            for region in regions:
                NegativeROA = region.attrib['NegativeROA']
                if NegativeROA == "0": # NROA = 0 rep for none background
                    points = []
                    for point in region.findall('Vertices/Vertex'):
                        x = float(point.attrib['X'])
                        y = float(point.attrib['Y'])
                        points.append([x, y])
                    pts = np.asarray([points], dtype=np.int32)
                    cv2.fillPoly(img=mask, pts=pts, color=255)

    for i in range(1, w // patch_size - 1):
        for j in range(1, h // patch_size - 1):
            # data checking: the data img should contain enough purple stuff
            img_patch = slide.read_region(
                (i * patch_size, j * patch_size),
                0,
                (patch_size, patch_size)
            )
            patch = img_patch.convert('RGB')
            img_single = patch.resize((1, 1), Image.ANTIALIAS)
            r, g, b = img_single.getpixel((0, 0))

            # mask checking: the mask should be alllllll black
            mask_patch = mask[j * patch_size:(j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
            if r < 220 and g < 220 and b < 220 and r > 100 and b > 30 and r > g + 20 and mask_patch.mean(axis=0).mean(axis=0) < 1:
                # save data
                patch = patch.resize((384, 384), Image.ANTIALIAS)
                save_file(patch, save_root + "\\data\\" + str(type) + '\\' + img_name + '-' + str(i) + '-' + str(j))
                # save mask
                mask_patch = cv2.resize(mask_patch, (384, 384))
                cv2.imwrite(save_root + "\\mask\\" + str(type) + '\\' + img_name + '-' + str(i) + '-' + str(j) + '.jpg',
                            mask_patch)


"""def xml2mask_check(img_name, data_root, patch_size = 2560):
    img = os.path.join(data_root, img_name + '.svs')
    print(img)
    xml = os.path.join(data_root, img_name + '.xml')

    slide = openslide.open_slide(img)
    w, h = slide.level_dimensions[0]
    mask = np.zeros([h, w], dtype=np.uint8)

    tree = ET.parse(xml)
    root = tree.getroot()
    Annotations = root.findall('Annotation')
    for Annotation in Annotations:
        id = Annotation.attrib['Id']
        if id == "2":
            regions = Annotation.findall('Regions/Region')
            for region in regions:
                NegativeROA = region.attrib['NegativeROA']
                if NegativeROA == "0":
                    points = []
                    for point in region.findall('Vertices/Vertex'):
                        x = float(point.attrib['X'])
                        y = float(point.attrib['Y'])
                        points.append([x, y])
                    pts = np.asarray([points], dtype=np.int32)
                    cv2.fillPoly(img=mask, pts=pts, color=255)
    cv2.imwrite(data_root + "\\mask\\" + img_name + '.jpg', mask)"""



def main():
    data_root = r'I:\Puzzle_Tuning_Datasets\Raw\PAIP2019\Training_phase_2'
    save_root = r'I:\Puzzle_Tuning_Datasets\Raw\PAIP2019'

    make_and_clear_path(os.path.join(save_root, 'mask\\pos'))
    make_and_clear_path(os.path.join(save_root, 'data\\pos'))
    make_and_clear_path(os.path.join(save_root, 'mask\\neg'))
    make_and_clear_path(os.path.join(save_root, 'data\\neg'))

    all_Annotations = find_all_files(data_root, 'xml')
    for Annotation in all_Annotations:
        img_name = os.path.split(Annotation)[1].split('.')[0]
        xml2mask_neg(img_name, data_root, save_root, type='neg')


    # cv2.imwrite(r'G:\Puzzle_WSI\HER2 tumor ROIs_v3\pkg_v3\Yale_HER2_cohort\mask\Her2Neg_Case_02' + ".jpg", mask)


if __name__ == '__main__':
    main()
