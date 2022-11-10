"""
'JPG_cropping'  ver： 22 Nov 10
Crop pathology images into patches  Using average filtering to screen the useful pieces which are mostly red/purple

"""


import os
import shutil
import PIL.Image as Image
import numpy as np
import torch
from torchvision import transforms
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
    factor = min(w//patch_size[0], h//patch_size[1])
    numpy_img = img.crop([0, 0, factor*patch_size[0], factor*patch_size[1]])
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
        print(x.shape)
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


def cut_to_patch(numpy_img, save_root, patch_size, img_name):
    img_split = to_patch(patch_size)
    img_patches = img_split(numpy_img)
    i = 0
    for patch in img_patches:
        patch = patch.permute(1, 2, 0)
        patch = patch.numpy()
        patch = Image.fromarray(patch.astype('uint8')).convert('RGB')
        save_file(patch, os.path.join(save_root, img_name + str(i)))
        i = i+1


def read_and_convert(data_root, save_root, suffix=None, patch_size=(256, 256)):
    # 一次处理只一个数据集, 每个数据集的处理方式可能有不同

    # 读入所有数据
    all_files = find_all_files(data_root, suffix)

    # 把所有数据转换为同一个格式
    for img in all_files:
        numpy_img = convert_to_npy(img, patch_size)
        img_name = os.path.split(img)[1].split('.')[0]
        cut_to_patch(numpy_img, save_root, patch_size, img_name)


if __name__ == '__main__':
    read_and_convert(r'/Users/munros/MIL/Processed/Colonoscopy/tissue-train-pos/data/Positive',
                     r'/Users/munros/PuzzleTuning/Colonoscopy_patch',
                     'jpg')