from tkinter import Tk
from tkinter import *


class PfTk(Tk):
    """
    Tk的扩展
    增加了对鼠标左键拖动的支持
    封装了部分功能方法
    """

    def __init__(self, title='PfTk', width=600, height=300, bg='red'):
        """
        初始化Tk
        :param title: str: 窗口标签(默认: PfTk)
        :param width: int: 宽度(默认: 300)
        :param height: height: 高度(默认: 300)
        :param bg: str: 背景色(默认: white)
        """
        # 调用父类构造方法
        super().__init__()
        # 设置窗口大小与居中
        self.geometry(f'{width}x{height}+'
                      f'{int((self.winfo_screenwidth() - width) / 2)}+'
                      f'{int((self.winfo_screenheight() - height) / 2)}')
        # 标题
        self.title(title)
        # 背景色
        self.config(bg=bg)
        # 鼠标拖动
        self.__motion = False
        # 鼠标点击坐标
        self.__clickX, self.__clickY = 0, 0
        # 绑定鼠标点击事件
        self.bind("<Button-1>", self.__onClick)
        # 绑定鼠标拖动事件
        self.bind('<B1-Motion>', self.__onMotion)
        self.initWidgets()

    def initWidgets(self):
        self.button_sync = Button(self.master, text='测试按钮', command=self.sync_file)
        self.button_sync.pack()

    def sync_file(self):
        print("sync_file")

    def bd(self, tf):
        """
        设置窗口边框
        :param tf: bool: 是否带有边框(默认: True)
        :return:
        """
        self.overrideredirect(not tf)

    def topmost(self, tf):
        """
        设置窗口置顶
        :param tf: bool: 是否置顶(默认: False)
        :return:
        """
        self.wm_attributes("-topmost", tf)

    def alpha(self, f):
        """
        设置窗口透明度
        :param f: float: 透明度(默认: 1.0)
        :return:
        """
        self.attributes('-alpha', f)

    def colorAlpha(self, color):
        """
        设置窗口某一颜色完全透明
        :param color: str: 颜色
        :return:
        """
        self.attributes('-transparentcolor', color)

    def move(self, tf):
        """
        设置窗口拖动
        :param tf: bool: 是否可以拖动(默认: False)
        :return:
        """
        self.__motion = tf

    def disable(self, tf):
        """
        设置窗口功能
        :param tf: bool: 是否具备功能(默认: True)
        :return:
        """
        self.attributes("-disabled", tf)

    def show(self, tf):
        """
        设置窗口显示
        :param tf: 是否显示(默认: True)
        :return:
        """
        if tf:
            self.deiconify()
        else:
            self.withdraw()

    def __onClick(self, event):
        """
        窗口鼠标左键点击事件
        :param event: obj: 鼠标左键点击事件
        :return:
        """
        print("记录鼠标左键点击坐标")
        print(event)
        # 记录鼠标左键点击坐标
        self.__clickX = event.x
        self.__clickY = event.y

    def __onMotion(self, event):
        """
        窗口鼠标拖动事件
        :param event: obj: 鼠标拖动事件
        :return:
        """
        print(event)
        if self.__motion:
            self.geometry(f'{self.winfo_width()}x{self.winfo_height()}+'
                          f'{int(self.winfo_x() + event.x - self.__clickX)}+'
                          f'{int(self.winfo_y() + event.y - self.__clickY)}')


# root = Tk()
PfTk().mainloop()

# root.mainloop()