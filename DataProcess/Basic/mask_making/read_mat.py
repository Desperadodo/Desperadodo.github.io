import glob
import cv2
import numpy as np
from scipy.io import loadmat


# 数据矩阵转图片的函数
def MatrixToImage(data):
    data = (data != 0)*255
    new_im = data.astype(np.uint8)
    return new_im



file = r'E:\A_bunch_of_data\Raw\pRCC\nuclei_prediction\TCGA-2K-A9WE-01Z-00-DX1.ED8ADE3B-D49B-403B-B4EB-BD11D91DD676\80052_52517_2000.mat'

m = loadmat(file)
daaa = m.keys()

data=m['Type']

print(data)

new_im = MatrixToImage(data)
cv2.imshow("asdf",new_im)
cv2.imwrite("E:\A_bunch_of_data\Raw\TCGA-2K-A9WEE.jpg", new_im)
cv2.waitKey()

    # print(data)