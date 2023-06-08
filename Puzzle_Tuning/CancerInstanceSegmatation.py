import numpy as np
import scipy.misc
from PIL import Image
import os
from matplotlib import pyplot as plt

"""path = 'npy文件路径/'
dir = os.listdir(path)
for a in dir:
    stem , suffix =os.path.splitext(a)
    imgs_test = np.load(str(path)+str(a),allow_pickle=True) #读入.npy文件
    data = imgs_test.item()
    #print(data)
    cam = data['cam']
    img = tensor_to_np(cam)
    im = Image.fromarray(img)

    im.save('保存路径/'+stem+'.jpg')"""

path = r'F:\Puzzle Tuning Datasets\archive（Cancer Instance Segmentation and Classification 1）\hovernet_format\hovernet_format\train\train\0.npy' # 要转换为图片的.npy文件
data = np.load(path)
image = Image.fromarray(data)
image.show()