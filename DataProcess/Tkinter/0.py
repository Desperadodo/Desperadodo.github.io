import os
os.add_dll_directory(r"D:\chrome_download\github220901\openslide-win64\bin")
import openslide
import os
from PIL import Image

data_path0 =r"F:\MIL_datasets\CAMELYON16\training\mask"
tif_path0 =r"F:\MIL_datasets\CAMELYON16\training\mask_tiffff"
save_path0 =r"F:\MIL_datasets\CAMELYON16\training\mask_jpgggg"
level = 4
del_list = os.listdir(data_path0)
for f in del_list:
    image_path = os.path.join(data_path0, f)
    tif_path = os.path.join(tif_path0, f[:-4] + '.tif')
    target_path = os.path.join(save_path0, f[:-4] + '.jpg')
    with openslide.OpenSlide(image_path) as slide:
        new_size = slide.level_dimensions[level]
        new_image = slide.read_region((0,0), level, new_size)
        new_image.save(tif_path)
        img = Image.open(tif_path)
        img = img.convert("RGB")
        img.save(target_path)

print("successfully!")