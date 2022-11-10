"""
Instant manual cropping  ver: Sep 5 official release
2.0.0: 增加了WSI的处理能力
"""
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from ctypes import *
import cv2
import numpy as np
import pyautogui as pag
from PIL import Image, ImageTk
import os
import subprocess
import numpy
os.add_dll_directory(r"D:\chrome_download\github220901\openslide-win64\bin")
import openslide


Image.MAX_IMAGE_PIXELS = None

__version__ = "2.0.0"


class _PointAPI(Structure):  # 用于getpos()中API函数的调用
    _fields_ = [("x", c_ulong), ("y", c_ulong)]


def xpos(): return pag.position()[0]  # pyautogui中的实时指针位置


def ypos(): return pag.position()[1]


# tkinter控件支持作为字典键。
# bound的键是dragger, 值是包含1个或多个绑定事件的列表, 值用于存储控件绑定的数据
# 列表的一项是对应tkwidget和其他信息的元组
bound = {}


def __add(wid, data):  # 添加绑定数据
    bound[wid] = bound.get(wid, []) + [data]


def __remove(wid, key):  # 用于从bound中移除绑定
    for i in range(len(bound[wid])):
        try:
            if bound[wid][i][0] == key:
                del bound[wid][i]
        except IndexError:
            pass


def __get(wid, key=''):  # 用于从bound中获取绑定数据
    if not key:
        return bound[wid][0]
    if key == 'resize':
        for i in range(len(bound[wid])):
            for s in 'nwse':
                if s in bound[wid][i][0].lower():
                    return bound[wid][i]
    for i in range(len(bound[wid])):
        if bound[wid][i][0] == key:
            return bound[wid][i]


def move(widget, x=None, y=None, width=None, height=None):
    """移动控件或窗口widget, 参数皆可选。"""
    x = x if x != None else widget.winfo_x()
    y = y if y != None else widget.winfo_y()
    width = width if width != None else widget.winfo_width()
    height = height if height != None else widget.winfo_height()
    if isinstance(widget, tk.Wm):
        widget.geometry("%dx%d+%d+%d" % (width, height, x, y))
    else:
        widget.place(x=x, y=y, width=width, height=height)
    return x, y, width, height


def _mousedown(event):
    if event.widget not in bound: return
    lst = bound[event.widget]
    for data in lst:  # 开始拖动时, 在每一个控件记录位置和控件尺寸
        widget = data[1]
        widget.mousex, widget.mousey = pag.position()
        widget.startx, widget.starty = widget.winfo_x(), widget.winfo_y()
        widget.start_w = widget.winfo_width()
        widget.start_h = widget.winfo_height()


def _drag(event):
    if event.widget not in bound: return
    lst = bound[event.widget]
    for data in lst:  # 多个绑定
        if data[0] != 'drag': return
        widget = data[1]
        dx = xpos() - widget.mousex  # 计算鼠标当前位置和开始拖动时位置的差距
        # 注: 鼠标位置不能用event.x和event.y
        # event.x,event.y与控件的位置、大小有关，不能真实地反映鼠标移动的距离差值
        dy = ypos() - widget.mousey
        move(widget, widget.startx + dx if data[2] else None,
             widget.starty + dy if data[3] else None)


"""def _resize(event):
    data = __get(event.widget, 'resize')
    if data is None: return
    widget = data[1]
    dx = event.x - widget.mousex  # 计算位置差
    dy = event.y - widget.mousey

    type = data[0].lower()
    minw, minh = data[2:4]
    if 's' in type:
        move(widget, height=max(widget.start_h + dy, minh))
    elif 'n' in type:
        move(widget, y=min(widget.starty + dy, widget.starty + widget.start_h - minh),
             height=max(widget.start_h - dy, minh))

    __remove(event.widget, data[0])  # 取消绑定, 为防止widget.update()中产生新的事件, 避免_resize()被tkinter反复调用
    widget.update()  # 刷新控件, 使以下左右缩放时, winfo_height()返回的是新的控件坐标, 而不是旧的
    __add(event.widget, data)  # 重新绑定

    if 'e' in type:
        move(widget, width=max(widget.start_w + dx, minw))
    elif 'w' in type:
        move(widget, x=min(widget.startx + dx, widget.startx + widget.start_w - minw),
             width=max(widget.start_w - dx, minw))"""


def draggable(tkwidget, x=True, y=True):
    """调用draggable(tkwidget) 使tkwidget可拖动。
tkwidget: 一个控件(Widget)或一个窗口(Wm)。
x 和 y: 只允许改变x坐标或y坐标。"""
    bind_drag(tkwidget, tkwidget, x, y)


def bind_drag(tkwidget, dragger, x=True, y=True):
    """绑定拖曳事件。
tkwidget: 被拖动的控件或窗口,
dragger: 接收鼠标事件的控件,
调用bind_drag后,当鼠标在dragger上拖动时, tkwidget会被拖动, 但dragger
作为接收鼠标事件的控件, 位置不会改变。
x 和 y: 同draggable()函数。"""
    dragger.bind("<Button-1>", _mousedown, add='+')
    dragger.bind("<B1-Motion>", _drag, add='+')
    __add(dragger, ('drag', tkwidget, x, y))  # 在bound字典中记录数据


"""def bind_resize(tkwidget, dragger, anchor, min_w=0, min_h=0, move_dragger=True):
    绑定缩放事件。
anchor: 缩放"手柄"的方位, 取值为N,S,W,E,NW,NE,SW,SE,分别表示东、西、南、北。
min_w,min_h: 该方向tkwidget缩放的最小宽度(或高度)。
move_dragger: 缩放时是否移动dragger。
其他说明同bind_drag函数。
    dragger.bind("<Button-1>", _mousedown, add='+')
    dragger.bind("<B1-Motion>", _resize, add='+')
    data = (anchor, tkwidget, min_w, min_h, move_dragger)
    __add(dragger, data)"""


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    在这个任务中，需要只寻找名称中带有'mask'的图片
    """
    res = []
    img_dict = {}
    i = 0
    for root, _, files in os.walk(root):
        # print(files)
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            f1 = f.split('.')[0]
            """if f1.endswith('mask') == 0:
                continue"""
            res.append(os.path.join(root, f))
            # print(os.path.join(root, f))
    res.sort()
    for dir in res:
        img_dict[i] = str(dir)
        # print(img_dict[i])
        i = i + 1

    return img_dict, i


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)
    # cv2.imwrite(save_dir + suffix, f_image)


def resize(w, h, w_box, h_box, pil_image):
    """
    对一个PIL_Image图像进行缩放，让它还在一个矩形框里保持比例
    """
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    # print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


count = 0
shot = 0
characteristic_length = 600


def test(root_from_1, root_from_2,
         root_to_1, root_to_2, root_to_3, root_to_4, root_to_5, root_to_6,
         root_WSI_1 = None, root_WSI_2 = None, WSI_level = None):
    btns = []  # 用btns列表存储创建的按钮
    cons_btns = []

    if root_WSI_1 is None:
        WSI = False
    else:
        WSI = True

    list = find_all_files(root_from_1, '.jpg')[0]
    max = find_all_files(root_from_1, '.jpg')[1]
    print(list)

    def remap_img(num, x_1, y_1, x_2, y_2, root_2 = False, black_mask = False, root_WSI = None, level = None, change_mask = False):
        if WSI is None:
            if not black_mask:
                if not root_2:
                    img = Image.open(list[num])
                else:
                    img = Image.open(os.path.join(root_from_2, os.path.split(list[num])[1]))
                img_cropped = img.crop([x_1, y_1, x_2, y_2])
            else:
                img_cropped = numpy.zeros(((x_2 - x_1), (y_2 - y_1)), dtype=numpy.uint8)
                img_cropped = Image.fromarray(img_cropped)
        else:
            if not black_mask:
                if not change_mask:
                    img_path_WSI = os.path.join(root_WSI, os.path.split(list[num])[1].split('.')[0] + ".tif")

                    slide = openslide.open_slide(img_path_WSI)

                    img_cropped = slide.read_region([x_1*16, y_1*16], 0, [(y_2 - y_1)*16, (x_2 - x_1)*16])
                    img_cropped = img_cropped.convert('RGB')
                else:
                    img_path_WSI = os.path.join(root_WSI, os.path.split(list[num])[1].split('.')[0] + ".tif")
                    slide = openslide.open_slide(img_path_WSI)
                    img_cropped = slide.read_region([x_1 * 16, y_1 * 16], 0, [(y_2 - y_1) * 16, (x_2 - x_1) * 16])
                    img_cropped = img_cropped.convert('RGB')
                    f_img_cropped = np.array(img_cropped)
                    img_cropped = Image.fromarray(np.uint8(f_img_cropped*255))
            else:

                img_cropped = numpy.zeros([(x_2 - x_1)*level, (y_2 - y_1)*level], dtype=numpy.uint8)
                img_cropped = Image.fromarray(img_cropped)
        return img_cropped

    def add_button(func, w, h, num):
        # func的作用是计算按钮新坐标
        b = tk.Button(root, background="red")
        b._func = func
        # bind_resize(btn,b,anchor)
        x, y = func()
        b.place(x=x, y=y, width=w, height=h)
        # b.bind('<B1-Motion>',adjust_button,add='+')
        # print(w, h, num)
        btns.append(b)

    def add_cons_button(func, w, h, num):
        b = tk.Button(root, background="red")
        b._func = func
        # bind_resize(btn,b,anchor)
        x, y = func()
        b.place(x=x, y=y, width=w, height=h)
        # b.bind('<B1-Motion>',adjust_button,add='+')
        # print(w, h, num)
        cons_btns.append(b)

    def adjust_button(event=None):
        # 改变大小或拖动后,调整手柄位置
        i = 0
        for b in btns:
            i += 1
            if i >= 10:
                break
            else:
                x, y = b._func()
                b.place(x=x, y=y)

    root = tk.Tk()
    root.title("CropMe")
    root.geometry('1500x1300')
    w_box = 1500
    h_box = 900

    img = Image.open(list[0])
    w, h = img.size
    img_resized = resize(w, h, w_box, h_box, img)
    tk_img = ImageTk.PhotoImage(img_resized)
    label = tk.Label(root, image=tk_img, width=w_box, height=h_box)
    label.grid(row=0, column=0, columnspan=5)

    entry_1 = tk.Entry(root)
    entry_1.grid(row=2, column=0)
    # characteristic_length = int(float(characteristic_length))

    entry_2 = tk.Entry(root)
    entry_2.grid(row=2, column=2)

    def get_characteristic_length(default=600):
        global characteristic_length
        string = entry_1.get()
        if string.isdigit() is True:
            characteristic_length = int(string)
        else:
            characteristic_length = default

        set_box(count)

    def get_count(default=0):
        global count
        string = entry_2.get()
        if string.isdigit() is True:
            count = int(string)
        else:
            count = 0
        global shot
        shot = 0
        # print(img_dict)
        img = Image.open(list[count])
        print(count, list[count])
        w, h = img.size
        img_resized = resize(w, h, w_box, h_box, img)
        tk_img = ImageTk.PhotoImage(img_resized)
        # label = tk.Label(root, image=tk_img, width=w_box, height=h_box)
        label.configure(image=tk_img)
        label.image = tk_img
        set_box(count)

    x1 = 400
    y1 = 300
    btn_size = 20
    btn = tk.Button(root, background='#ECf5EF')
    draggable(btn)
    btn.bind('<B1-Motion>', adjust_button, add='+')
    btn.place(x=x1, y=y1, width=btn_size, height=btn_size)

    def get_size(num):
        img = Image.open(list[num])
        w, h = img.size
        f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        resized_length = characteristic_length * factor
        b = 2
        c = 1
        a = resized_length - b
        return factor, resized_length, a, b, c, w

    def set_box(num):
        for b in btns:
            b.place_forget()
        del btns[:]  # 防止出现野按钮
        for b in cons_btns:
            b.place_forget()
        del cons_btns[:]  # 防止出现野按钮

        _, resized_length, a, b, c, w = get_size(num)

        # 创建各个手柄, 这里是控件缩放的算法
        add_button(lambda: (btn.winfo_x() - (resized_length / 2 - btn_size / 2),
                            btn.winfo_y() - (resized_length / 2 - btn_size / 2)), a, b, num)  # 上面的

        add_button(lambda: (btn.winfo_x() + (resized_length / 2 - b + btn_size / 2),
                            btn.winfo_y() - (resized_length / 2 - btn_size / 2)), b, a, num)  # 右边的

        add_button(lambda: (btn.winfo_x() - (resized_length / 2 - btn_size / 2),
                            btn.winfo_y() - (resized_length / 2 - btn_size / 2 - b)), b, a, num)  # 左边的

        add_button(lambda: (btn.winfo_x() - (resized_length / 2 - btn_size / 2 - b),
                            btn.winfo_y() + (resized_length / 2 - b + btn_size / 2)), a, b, num)  # 下面的

        root.update()

    # img_dict = find_all_files(r'/Users/munros/MIL/Processed/Colonoscopy/mask/Positive', '.jpg')
    # set_box(0)

    def shot_box(num):
        global shot
        shot += 1  # 每个小图的编号 在一次上或者下后会自动归零
        factor, resized_length, a, b, c, w = get_size(num)
        l = (1500 - w * factor) / 2
        x_1 = btn.winfo_x() - (resized_length / 2 - btn_size / 2) - l  # 边框尺寸
        x_2 = btn.winfo_x() + (resized_length / 2 - b + btn_size / 2) - l
        y_1 = btn.winfo_y() - (resized_length / 2 - btn_size / 2)
        y_2 = btn.winfo_y() + (resized_length / 2 - b + btn_size / 2)
        # shot完之后留下固定的框，线条细一点点
        add_cons_button(lambda: (btn.winfo_x() - (resized_length / 2 - btn_size / 2),
                                 btn.winfo_y() - (resized_length / 2 - btn_size / 2)), a, c, num)  # 上面的

        add_cons_button(lambda: (btn.winfo_x() + (resized_length / 2 - c + btn_size / 2),
                                 btn.winfo_y() - (resized_length / 2 - btn_size / 2)), c, a, num)  # 右边的

        add_cons_button(lambda: (btn.winfo_x() - (resized_length / 2 - btn_size / 2),
                                 btn.winfo_y() - (resized_length / 2 - btn_size / 2 - c)), c, a, num)  # 左边的

        add_cons_button(lambda: (btn.winfo_x() - (resized_length / 2 - btn_size / 2 - c),
                                 btn.winfo_y() + (resized_length / 2 - c + btn_size / 2)), a, c, num)  # 下面的

        x_1_r = int(x_1 / factor)
        y_1_r = int(y_1 / factor)
        x_2_r = x_1_r + characteristic_length
        y_2_r = y_1_r + characteristic_length

        img_name = os.path.split(list[num])[1].split('.')[0]
        img_name_s = img_name + '-' + str(characteristic_length)
        img_name_s = img_name_s + '-' + str(shot)

        print(x_1_r, y_1_r, x_2_r, y_2_r)

        img_1_cropped = remap_img(num, x_1_r, y_1_r, x_2_r, y_2_r, root_WSI=root_WSI_1, level=WSI_level, change_mask=True)
        if root_from_2 is None:
            img_1_cropped = remap_img(num, x_1_r, y_1_r, x_2_r, y_2_r, root_WSI=root_WSI_1, level=WSI_level,
                                      change_mask=False)
            img_2_cropped = remap_img(num, x_1_r, y_1_r, x_2_r, y_2_r, root_2=False, black_mask=True, level=WSI_level)
        else:
            img_1_cropped = remap_img(num, x_1_r, y_1_r, x_2_r, y_2_r, root_WSI=root_WSI_1, level=WSI_level,
                                      change_mask=True)
            img_2_cropped = remap_img(num, x_1_r, y_1_r, x_2_r, y_2_r, root_2=True, black_mask=False,
                                      root_WSI=root_WSI_2, level=WSI_level)
        save_dir_1 = os.path.join(root_to_1, img_name_s)
        save_dir_2 = os.path.join(root_to_2, img_name_s)
        save_file(img_1_cropped, save_dir_1)
        save_file(img_2_cropped, save_dir_2)

        # 中倍数：
        # shot_box_spilt(5, 3, img_1_cropped, img_2_cropped, img_name, root_to_3, root_to_4, shot)
        # 高倍数：
        # shot_box_spilt(10, 4, img_1_cropped, img_2_cropped, img_name, root_to_5, root_to_6, shot)

    def shot_box_spilt(a, b, img_1_cropped, img_2_cropped, img_name, root_to_1, root_to_2, shot):
        """
        :param img_name: 图片的原始名称
        :param root_to_1: 裁剪后的去向
        :param img_1_cropped: shotbox中裁剪好的低倍图片
        :param a: 将shotbox划分成a*a份
        :param b: 取中间的b*b份 注意a-b为2的倍数
        :return:
        """
        w, h = img_1_cropped.size
        start = w * ((a - b) / 2) / a
        stride = w / a

        split = 1

        for j in range(b):
            for i in range(b):
                x_1 = start + i * stride
                x_2 = x_1 + stride
                y_1 = start + j * stride
                y_2 = y_1 + stride
                img_name_s = img_name + '-' + str(int(characteristic_length / a))
                img_name_s = img_name_s + '-' + str(shot)
                img_name_s = img_name_s + '-' + str(split)
                split += 1

                img_1_split = img_1_cropped.crop([x_1, y_1, x_2, y_2])
                img_2_split = img_2_cropped.crop([x_1, y_1, x_2, y_2])
                save_dir_1 = os.path.join(root_to_1, img_name_s)
                save_dir_2 = os.path.join(root_to_2, img_name_s)
                save_file(img_1_split, save_dir_1)
                save_file(img_2_split, save_dir_2)

    def down():
        global count
        count += 1
        if count > max - 1:
            count = 0
        global shot
        shot = 0
        # print(img_dict)
        img = Image.open(list[count])
        print(count, list[count])
        w, h = img.size
        img_resized = resize(w, h, w_box, h_box, img)
        tk_img = ImageTk.PhotoImage(img_resized)
        # label = tk.Label(root, image=tk_img, width=w_box, height=h_box)
        label.configure(image=tk_img)
        label.image = tk_img
        set_box(count)

    def up():
        global count
        count -= 1
        if count < 1:
            count = max - 1
        global shot
        shot = 0
        # print(img_dict)
        img = Image.open(list[count])
        print(count, list[count])
        w, h = img.size
        img_resized = resize(w, h, w_box, h_box, img)
        tk_img = ImageTk.PhotoImage(img_resized)
        # label = tk.Label(root, image=tk_img, width=w_box, height=h_box)
        label.configure(image=tk_img)
        label.image = tk_img
        set_box(count)

    btn_up = tk.Button(root, text="上一张", command=lambda: up())
    btn_up.grid(row=1, column=0)
    btn_down = ttk.Button(root, text="下一张", command=lambda: down())
    btn_down.grid(row=1, column=3)
    btn_shot = tk.Button()
    btn_shot.invoke()
    btn_shot = tk.Button(root, text="    剪切    ", command=lambda: shot_box(count))
    btn_shot.grid(row=1, column=1)
    root.bind('<Return>', lambda event=None: btn_shot.invoke())
    btn_characteristic_length = ttk.Button(root, text="确认边长", command=lambda: get_characteristic_length())
    btn_characteristic_length.grid(row=2, column=1)
    btn_characteristic_length = ttk.Button(root, text="确认跳转", command=lambda: get_count())
    btn_characteristic_length.grid(row=2, column=3)



    root.mainloop()


if __name__ == "__main__": test(
    root_from_1=r'F:\MIL_datasets\CAMELYON16\training\normal_jpg',
    # root_from_2=r'F:\MIL_datasets\CAMELYON16\training\tumor_jpg',
    root_from_2= None,
    root_to_1=r"F:\MIL_datasets\CAMELYON16\training\cropped\data\normal",
    root_to_2=r"F:\MIL_datasets\CAMELYON16\training\cropped\mask\normal",
    root_to_3=r"/Users/munros/MIL/Processed/Colonoscopy_Cropped/MID/data/Negative",
    root_to_4=r"/Users/munros/MIL/Processed/Colonoscopy_Cropped/MID/mask/Negative",
    root_to_5=r"/Users/munros/MIL/Processed/Colonoscopy_Cropped/HI/data/Negative",
    root_to_6=r"/Users/munros/MIL/Processed/Colonoscopy_Cropped/HI/mask/Negative",
    root_WSI_1=r"F:\MIL_datasets\CAMELYON16\training\normal",
    root_WSI_2=r"F:\MIL_datasets\CAMELYON16\training\normal" ,
    WSI_level=16
)

"""
    root_from_1，root_from_2是原图mask和data的文件夹路径，以1为主2为从。对于neg，root_from_1写data, root_from_2写None
    root_to: 1，2：低倍的mask/data，neg反过来
            3，4：中倍的，以此类推
            
    目前MID是将LO分为5*5取中间3*3，HI是LO分为10*10取中间4*4
    具体可以在391，392行更改
"""