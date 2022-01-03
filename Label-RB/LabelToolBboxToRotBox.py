# -*- coding：utf-8 -*-
"""
time:2021/8/9 19:10
author:Lance
organization: HIT
contact: QQ:261983626 , wechat:yuan261983626
——————————————————————————————
description：
$
——————————————————————————————
note：
$
"""
from fiftyone import ViewField as F
import fiftyone as fo
import cv2 as cv
import numpy as np
from enum import Enum
import sys
import os
from QTUI.labelIt import Ui_MainWindow
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QMainWindow
import PySide6.QtCore as QtCore
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap, QImage, QImageReader


class enum_软件状态(Enum):
    """
    仅用于提示，不用于判断
    """
    初始状态 = "初始化...",
    载入数据集 = "正在载入数据集...",
    数据集载入完成 = "数据集载入完成",
    载入数据集中 = "正在载入数据集",
    标记物品 = "请画出物品的部分区域",
    标记背景 = "请标记出确定是背景的区域",
    选择标记区域 = "请在原始图像中框出要标记的区域",
    完成区域标记 = "已完成区域标记，接下来请标注物品",
    没有数据集 = "没有加载数据集,请先加载数据集"


class enum_检查项(Enum):
    数据库名称 = 0
    图片路径 = 1


class enum_检查结果信息(Enum):
    """
    仅用于提示，不用于判断
    """
    数据库不存在 = "数据库不存在"
    没有找到路径下的图片 = "没有找到路径下的图片"
    默认 = "成功"


class enum_检查结果(Enum):
    成功 = 1
    失败 = 0


class enum_绘制状态(Enum):
    """
    表示当前正在绘制的对象
    """
    绘制前景 = 0
    绘制背景 = 1


class enum_控件对象(Enum):
    标注区域 = 0
    图片区域 = 1


class enum_求解模式(Enum):
    """
    表示手动绘制包围盒还是opencv分水岭方法求解包围盒
    """
    机器求解 = 0
    人工求解 = 1


class enum_颜色(Enum):
    红色 = (0, 0, 255)
    蓝色 = (255, 0, 0)
    黄色 = (0, 255, 0)


class enum_透明度(Enum):
    前景 = 0
    背景 = 50


class RotBoxLabel(object):
    def __init__(self, img_目标图像: np.ndarray, index_实例序号: int = 1, size_画笔尺寸: int = 10):
        self.Flag_正在绘制 = False
        self.ix = -1
        self.iy = -1
        self.img_目标图像 = img_目标图像.copy()  # 仅被读取
        self.img_交互显示图像 = img_目标图像.copy()  # 可被修改
        self.img_前景 = np.zeros(self.img_目标图像.shape, np.uint8)
        self.img_背景 = np.zeros(self.img_目标图像.shape, np.uint8)
        self.img_maker_分水岭模板 = np.zeros(self.img_目标图像.shape[0:2], np.int32)
        self.points_最新背景点: tuple = ()
        self.points_最新前景点: tuple = ()
        self.points_历史背景点: list = []
        self.points_历史前景点: list = []
        self.state_操作历史: list = []
        self.box_最小矩形框 = None  # [x_center,y_center,long_side, short_side,angle]
        self.box_最小包围框4点形式 = None
        self.bbox_包围框 = None
        self.pixel_实例像素点 = None  # 可以用作实例分割的标注
        self.mask_分水岭分割结果 = None  # 分水岭算法的分割结果
        self.mask_实例最小区域模板 = None  # 实例的最小区域模板
        self.mask_物品bbox蒙版 = None
        self.draw_绘制状态 = enum_绘制状态.绘制前景
        self.cal_求解模式 = enum_求解模式.机器求解
        self.size_画笔大小 = size_画笔尺寸
        self.color_前景画色 = enum_颜色.红色.value
        self.color_背景画色 = enum_颜色.蓝色.value
        self.index_实例序号 = index_实例序号  # 实例在同一张图片上的序号 从1开始 ，即同一个图片的第几个实例

    def draw_circle(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            if enum_绘制状态.绘制前景 == self.draw_绘制状态:
                cv.circle(self.img_交互显示图像, (x, y), self.size_画笔大小, self.color_前景画色, -1)
                self.points_最新前景点 += ((x, y),)
            elif enum_绘制状态.绘制背景 == self.draw_绘制状态:
                cv.circle(self.img_交互显示图像, (x, y), self.size_画笔大小, self.color_背景画色, -1)
                self.points_最新背景点 += ((x, y),)
        elif event == cv.EVENT_LBUTTONUP:
            if enum_绘制状态.绘制前景 == self.draw_绘制状态:
                self.points_历史前景点.append(self.points_最新前景点)
                self.points_最新前景点 = ()
            elif enum_绘制状态.绘制背景 == self.draw_绘制状态:
                self.points_历史背景点.append(self.points_最新背景点)
            self.points_最新背景点 = ()
            self.rbox_求解最小包围框()
            print(self.box_最小矩形框)
            print("求解最小包围框")

    def rbox_求解最小包围框(self):
        flag_是否有背景点 = any(self.points_历史背景点)
        flag_是否有前景点 = any(self.points_历史前景点)
        self.img_maker_分水岭模板 = np.zeros(self.img_目标图像.shape[0:2], np.uint8)
        area_背景点 = ()
        area_前景点 = ()
        kernel = np.ones((self.size_画笔大小, self.size_画笔大小), np.uint8)
        for points in self.points_历史前景点:
            area_前景点 += points
        for points in self.points_历史背景点:
            area_背景点 += points
        if flag_是否有前景点:
            index_前景索引 = tuple(np.array(area_前景点)[:, ::-1].T)
            img_前景 = self.img_maker_分水岭模板.copy()
            img_前景[index_前景索引] = 255
            img_前景 = cv.dilate(img_前景, kernel, iterations=3)
            img_前景 = cv.erode(img_前景, kernel, iterations=1)
            self.img_maker_分水岭模板[img_前景 == 255] = 2
        if flag_是否有背景点:
            index_背景索引 = tuple(np.array(area_背景点)[:, ::-1].T)
            img_背景 = self.img_maker_分水岭模板.copy()
            img_背景[index_背景索引] = 255
            img_背景 = cv.dilate(img_背景, kernel, iterations=3)
            img_背景 = cv.erode(img_背景, kernel, iterations=1)
            self.img_maker_分水岭模板[img_背景 == 255] = 1

        # ret, dst = cv.threshold(self.img_maker_分水岭模板, 1, 255, type)
        # self.img_maker_分水岭模板 = cv.dilate(self.img_maker_分水岭模板, kernel, iterations=2)  # 膨胀运算
        # cv.imshow('a', self.img_maker_分水岭模板)
        # cv.waitKey(0)
        if flag_是否有背景点 and flag_是否有前景点:
            self.mask_分水岭分割结果 = cv.watershed(self.img_目标图像, np.array(self.img_maker_分水岭模板, np.int32))
            #  2:实例    1：背景   -1：边界  0：什么都没有
        if not (self.mask_分水岭分割结果 is None):
            # b_通道, g_通道, r_通道 = cv.split(self.img_目标图像)
            # aph_通道 = np.ones(b_通道.shape, dtype=b_通道.dtype) * 255
            # aph_通道[self.box_最小矩形框 == 2] = enum_透明度.前景.value
            # aph_通道[self.box_最小矩形框 == 1] = enum_透明度.背景.value
            # self.img_交互显示图像 = cv.merge((self.img_交互显示图像, aph_通道))
            self.img_交互显示图像 = self.img_目标图像.copy()
            self.img_交互显示图像[:, :, 0][self.mask_分水岭分割结果 == 2] = 255
            # 求解最小包围框
            self.mask_实例最小区域模板 = np.zeros(self.img_目标图像.shape[0:2], np.uint8)
            self.mask_实例最小区域模板[self.mask_分水岭分割结果 == 2] = 1
            area_where = np.where(self.mask_分水岭分割结果 == 2)  # 实例的像素点位置
            self.pixel_实例像素点 = area_where
            #  [
            #  [x1,x2,x3,x4,x5...]
            #  [y1,y2,y3,y4,y5...]
            #  ]
            index_pixel_像素点索引 = np.vstack((area_where[1], area_where[0])).T
            rect_最小矩形框 = cv.minAreaRect(index_pixel_像素点索引)
            bbox_包围框 = cv.boundingRect(index_pixel_像素点索引)
            x, y, w, h = bbox_包围框[:]
            self.mask_物品bbox蒙版 = self.mask_实例最小区域模板[y:y + h,
                                 x:x + w]
            self.mask_物品bbox蒙版 = (self.mask_物品bbox蒙版[:] == 1)
            self.bbox_包围框 = bbox_包围框
            self.box_最小矩形框 = rect_最小矩形框
            self.box_最小包围框4点形式 = np.int0(cv.boxPoints(self.box_最小矩形框))
            self.img_交互显示图像 = self.drawing_绘制矩形框(self.img_交互显示图像, self.box_最小包围框4点形式)
            # x, y, w, h = self.bbox_包围框[:]
            # self.img_交互显示图像 = cv.rectangle(self.img_交互显示图像, (x, y), (x + w, y + h), enum_颜色.黄色.value, 2)

    def drawing_初始化绘制(self):
        if self.box_最小矩形框 is not None:
            self.box_最小包围框4点形式 = np.int0(cv.boxPoints(self.box_最小矩形框))
            self.img_交互显示图像 = self.drawing_绘制矩形框(self.img_交互显示图像, self.box_最小包围框4点形式)
        if self.mask_物品bbox蒙版 is not None and self.box_最小矩形框 is not None:
            area_where = np.where(self.mask_物品bbox蒙版)  # 实例的像素点位置
            r_行 = area_where[0] + self.bbox_包围框[1]
            c_列 = area_where[1] + self.bbox_包围框[0]
            self.img_交互显示图像[:, :, 0][(r_行, c_列)] = 255

    @staticmethod
    def drawing_绘制矩形框(img_目标图片, rbox_旋转, thickness=1):
        pixel_点1 = tuple(rbox_旋转[0])
        pixel_点2 = tuple(rbox_旋转[1])
        pixel_点3 = tuple(rbox_旋转[2])
        pixel_点4 = tuple(rbox_旋转[3])
        img_目标图片 = cv.line(img_目标图片, pixel_点1, pixel_点2, (0, 0, 250), thickness=thickness)
        img_目标图片 = cv.line(img_目标图片, pixel_点2, pixel_点3, (0, 0, 250), thickness=thickness)
        img_目标图片 = cv.line(img_目标图片, pixel_点3, pixel_点4, (0, 0, 250), thickness=thickness)
        img_目标图片 = cv.line(img_目标图片, pixel_点4, pixel_点1, (0, 0, 250), thickness=thickness)
        return img_目标图片

    def trackbar_改变画笔尺寸(self, x):
        self.size_画笔大小 = x

    def refresh_重新绘制(self):
        self.points_最新背景点: tuple = ()
        self.points_最新前景点: tuple = ()
        self.points_历史背景点: list = []
        self.points_历史前景点: list = []
        self.state_操作历史: list = []
        self.box_最小矩形框 = None  # [x_center,y_center,long_side, short_side,angle]
        self.box_最小包围框4点形式 = None
        self.pixel_实例像素点 = None  # 可以用作实例分割的标注
        self.mask_分水岭分割结果 = None  # 分水岭算法的分割结果
        self.mask_实例最小区域模板 = None  # 实例的最小区域模板
        self.img_交互显示图像 = self.img_目标图像.copy()

    def running_运行求解(self):
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.draw_circle)
        cv.moveWindow('image', 300, 300)
        cv.createTrackbar('brush size', 'image', self.size_画笔大小, 30, self.trackbar_改变画笔尺寸)
        while 1:
            cv.imshow('image', self.img_交互显示图像)
            k = cv.waitKey(1) & 0xFF
            if k == ord('1'):
                self.draw_绘制状态 = enum_绘制状态.绘制前景
            elif k == ord('2'):
                self.draw_绘制状态 = enum_绘制状态.绘制背景
            elif k == ord("3"):
                self.refresh_重新绘制()
            elif k == ord("q"):
                break
        cv.destroyWindow('image')

    @staticmethod
    def display_显示图片(img):
        cv.imshow('a', img)
        cv.waitKey(0)
        cv.destroyWindow('a')


class LabelIt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        ################
        # 数据集相关
        ################
        self.dataset_目标数据集: fo.Dataset = None
        self.dataset_fiftyone数据集: fo.Dataset = None
        self.name_dataset_数据集名称: str = None
        self.number_总样本数: int = 0
        self.number_已标记样本数: int = None
        self.number_1例样本数: int = None
        self.number_2_5例样本数: int = None
        self.number_6_9例样本数: int = None
        self.number_10_13例样本数: int = None
        self.pth_图片文件路径: str = ""
        self.class_类别列表: list = []
        ################
        # 当前照片相关
        ################
        self.index_照片的索引: int = None
        self.sample_照片样本: fo.Sample = None
        self.img_照片原始数据: np.ndarray = None
        self.img_照片用于显示: QImage = None
        self.pth_当前图片路径: str = ""
        self.number_图片的物品数: int = None
        self.width_图片样本宽: int = None
        self.height_图片样本高: int = None
        self.width_图片缩放后的宽: int = None
        self.height_图片缩放后的高: int = None
        self.scan_图片缩放比: float = None
        self.thickness_画笔厚度: int = 10
        self.bbox_待标注的区域: list = []  # 统一是 bbox  (x,y,w,h)的格式 范围在[0 1]之间，和正框统一
        self.point_新实例的起始点: tuple = ()
        self.point_新实例的终点: tuple = ()
        self.thickness_scan_旋转框画笔比例 = 0.01
        self.thickness_scan_正框画笔比例 = 0.01

        ###############
        # 当前标注相关
        ###############
        self.detection_当前物品 = None
        self.index_照片中物品的索引: int = 0
        self.img_标注区域原始数据: np.ndarray = None
        self.img_标注区域数据用于显示: QImage = None
        self.label_区域标注体: RotBoxLabel = None
        self.width_标注区域宽: int = None
        self.height_标注区域高: int = None
        self.scan_标注区域缩放比: float = None
        self.size_画笔尺寸: int = 3
        ###############
        # 标注区域
        ###############
        self.width_物品缩放后的宽: int = None
        self.height_物品缩放后的高: int = None
        ###############
        # 标识符
        ###############
        self.Flag_显示正框: bool = False
        self.Flag_显示旋转框: bool = False
        self.Flag_显示分割结果: bool = False
        self.Flag_已有正框: bool = False
        self.Flag_标注单个实例: bool = False
        # 程序标志符
        self.Flag_是否已成功显示图片: bool = False
        self.Flag_是否已成功载入数据集: bool = False
        self.Flag_是否已有物品待标注: bool = False
        self.Flag_是否新建实例: bool = False  # 如果采用选取区域的方式的话。
        ###############
        # 文本显示内容
        ###############
        self.text_标签1: str = None
        self.text_标签2: str = None
        self.text_标签3: str = None
        ##############################
        ## 鼠标操作
        self.mouse_press_object_鼠标单击对象: enum_控件对象 = enum_控件对象.图片区域

        ##############
        # 颜色参数
        self.color_正框 = enum_颜色.黄色
        self.color_旋转框 = enum_颜色.红色
        self.color_实例分割 = enum_颜色.蓝色
        ##############

        self.timer = QTimer()
        self.timer.timeout.connect(self.fresh_state_更新显示状态)
        self.timer.start(30)

        self.rename_更换名称()
        self.connect_建立连接()

        self.ini_初始化界面()
        self.size_图片显示区域尺寸 = self.label_标签_显示图片.height()
        self.area_size_区域尺寸 = self.label_标签_显示物品.height()

    def rename_更换名称(self):
        ###########
        # 标签
        ###########
        self.label_标签_显示图片 = self.ui.label_image_ori
        self.label_标签_显示物品 = self.ui.label_image_object

        ###########
        # 按钮
        ###########
        # 数据集
        self.pushButton_按钮_导入并创建数据集 = self.ui.pushButton_create_dataset_from_image_dir
        self.pushButton_按钮_fiftyone导入 = self.ui.pushButton_load_dataset_from_fiftyone_dataset
        self.pushButton_按钮_删除数据集 = self.ui.pushButton_delete_dataset
        self.radioButton_单选按钮_是否从文件夹导入并创建数据集 = self.ui.radioButton_create_dataset_from_images_dir
        self.pushButton_按钮_导出数据集 = self.ui.pushButton_expand_the_dataset
        self.pushButton_按钮_选择图片目录 = self.ui.pushButton_image_dir_to_load
        self.pushButton_按钮_选择标签文件 = self.ui.pushButton_label_file_to_load
        self.pushButton_按钮_网页端预览数据集 = self.ui.pushButton_display_on_web
        self.pushButton_按钮_执行筛选 = self.ui.pushButton_select_running
        self.pushButton_按钮_刷新数据集 = self.ui.pushButton_update

        # 图片
        self.pushButton_按钮_上一图片 = self.ui.pushButton_front_picture
        self.pushButton_按钮_下一图片 = self.ui.pushButton_next_picture
        self.pushButton_按钮_保存图片 = self.ui.pushButton_save_picture
        self.pushButton_按钮_删除照片 = self.ui.pushButton_delete_image
        self.pushButton_按钮_载入指定图片 = self.ui.pushButton_load_special_image
        self.pushButton_按钮_保存新实例 = self.ui.pushButton_label_save_new_object
        self.radioButton_单选按钮_显示正框 = self.ui.radioButton_displaybbox
        self.radioButton_单选按钮_显示旋转框 = self.ui.radioButton_displayrbox
        self.radioButton_单选按钮_显示分割 = self.ui.radioButton_displaymask

        # 物品
        self.pushButton_按钮_标注背景 = self.ui.pushButton_brush_background
        self.pushButton_按钮_标注物品 = self.ui.pushButton_brush_object
        self.pushButton_按钮_下一物品 = self.ui.pushButton_next_object
        self.pushButton_按钮_上一物品 = self.ui.pushButton_front_object
        self.pushButton_按钮_重新标注 = self.ui.pushButton_refresh_object
        self.pushButton_按钮_保存物品 = self.ui.pushButton_save_object
        self.pushButton_按钮_删除实例 = self.ui.pushButton_delete_instance
        self.radioButton_单选按钮_锁定标签 = self.ui.radioButton_locklabel
        ###########
        # 文本
        ###########
        # 数据集
        self.itext_输入文本_导入图片的文件夹路径 = self.ui.textlabel_path_image_dir
        self.itext_输入文本_fiftyone名称 = self.ui.textlabel_dataset_name
        self.comboBox_下拉选项_数据集列表 = self.ui.comboBox_dataset_list
        self.comboBox_类别过滤 = self.ui.comboBox_class_filer
        self.itext_输入文本_路径_数据集标签文件目录 = self.ui.textlabel_path_label_file
        # 图片
        self.itext_输入文本_图片样本路径 = self.ui.textlabel_path_current_picture
        # 物品
        self.itext_输入文本_标签1 = self.ui.textlabel_label_name_1
        self.itext_输入文本_标签2 = self.ui.textlabel_label_name_2
        self.itext_输入文本_标签3 = self.ui.textlabel_label_name_3
        self.otext_输出文本_物品序号 = self.ui.textlabel_index_object
        # 处理进度
        self.otext_输出文本_样本总数 = self.ui.textBrowser_sum_images
        self.otext_输出文本_已标记样本 = self.ui.textBrowser_num_labeled
        self.otext_输出文本_1例数 = self.ui.textBrowser_num_1_object
        self.otext_输出文本_2_5例数 = self.ui.textBrowser_num_2_5_object
        self.otext_输出文本_6_9例数 = self.ui.textBrowser_num_6_9_object
        self.otext_输出文本_10_13例数 = self.ui.textBrowser_num_10_13_object

        self.otext_输出文本_当前样本序数 = self.ui.textBrowser_index_sample
        self.otext_输出文本_实例总数 = self.ui.textBrowser_number_of_objects_in_picture

        ###########
        # 其它
        ###########
        self.bar_处理进度条 = self.ui.progressBar_label  # 进度条
        self.slider_画笔尺寸滑块 = self.ui.horizontalSlider_brush_size
        self.label_view_输出文本_提示信息 = self.ui.label_state_feedback
        self.groupBox_图片显示区域群 = self.ui.groupBox_image_ori
        self.groupBox_标注物品区域群 = self.ui.groupBox_label_object

    def connect_建立连接(self):
        # 数据集
        self.pushButton_按钮_fiftyone导入.clicked.connect(self.load_载入FiftyOne数据集)
        self.pushButton_按钮_导入并创建数据集.clicked.connect(self.create_创建数据集)
        self.pushButton_按钮_删除数据集.clicked.connect(self.delete_删除数据集)
        self.pushButton_按钮_导出数据集.clicked.connect(self.export_导出数据集)
        self.pushButton_按钮_选择图片目录.clicked.connect(self.select_dir_选择图片目录)
        self.pushButton_按钮_选择标签文件.clicked.connect(self.select_file_选择标签文件)
        self.pushButton_按钮_执行筛选.clicked.connect(self.load_载入部分数据集)
        self.pushButton_按钮_刷新数据集.clicked.connect(self.update_刷新数据集)
        self.pushButton_按钮_网页端预览数据集.clicked.connect(self.display_网页端显示)
        # self.radioButton_单选按钮_已有正框.clicked.connect(self.radioButton_单选按钮_已有正框)
        # self.radioButton_单选按钮_标注单个实例.clicked.connect(self.radioButton_单选按钮_标注单个实例)
        # 图片
        self.pushButton_按钮_上一图片.clicked.connect(self.front_上一张图片)
        self.pushButton_按钮_下一图片.clicked.connect(self.next_下一张图片)
        self.pushButton_按钮_删除照片.clicked.connect(self.delete_删除这张照片)
        self.pushButton_按钮_保存图片.clicked.connect(self.save_保存图片)
        self.pushButton_按钮_保存新实例.clicked.connect(self.save_保存新建的物品)
        self.pushButton_按钮_载入指定图片.clicked.connect(self.read_载入指定图片)
        # 物品
        self.pushButton_按钮_重新标注.clicked.connect(self.refresh_重新标注物品)
        self.pushButton_按钮_上一物品.clicked.connect(self.front_上一个物品)
        self.pushButton_按钮_下一物品.clicked.connect(self.next_下一个物品)
        self.pushButton_按钮_标注背景.clicked.connect(self.label_标注背景区域)
        self.pushButton_按钮_标注物品.clicked.connect(self.label_标注物品区域)
        self.pushButton_按钮_保存物品.clicked.connect(self.save_object_保存物品)
        self.pushButton_按钮_删除实例.clicked.connect(self.delete_删除当前实例)

        # 其它
        self.slider_画笔尺寸滑块.valueChanged.connect(self.change_更改画笔尺寸)
        self.radioButton_单选按钮_显示旋转框.clicked.connect(self.drawing_绘制原始图片)
        self.radioButton_单选按钮_显示正框.clicked.connect(self.drawing_绘制原始图片)
        self.radioButton_单选按钮_显示分割.clicked.connect(self.drawing_绘制原始图片)
        # self.label_标签_显示物品.mouseMoveEvent(self.mouseMoveEvent)

    def ini_初始化界面(self):
        self.comboBox_下拉选项_数据集列表.clear()
        dataset_list = fo.list_datasets()
        for one_dataset in dataset_list:
            self.comboBox_下拉选项_数据集列表.addItem(one_dataset)

    def fresh_state_更新显示状态(self):
        if str(self.number_总样本数):               self.otext_输出文本_样本总数.setText(str(self.number_总样本数))
        # if str(self.pth_当前图片路径):               self.itext_输入文本_图片样本路径.setText(str(self.pth_当前图片路径))
        if str(self.number_1例样本数):              self.otext_输出文本_1例数.setText(str(self.number_1例样本数))
        if str(self.number_2_5例样本数):            self.otext_输出文本_2_5例数.setText(str(self.number_2_5例样本数))
        if str(self.number_6_9例样本数):            self.otext_输出文本_6_9例数.setText(str(self.number_6_9例样本数))
        if str(self.number_10_13例样本数):          self.otext_输出文本_10_13例数.setText(str(self.number_10_13例样本数))
        # if str(self.text_标签1):                   self.itext_输入文本_标签1.setText(self.text_标签1)
        # if str(self.text_标签2):                   self.itext_输入文本_标签2.setText(self.text_标签2)
        # if str(self.text_标签3):                   self.itext_输入文本_标签3.setText(self.text_标签3)
        if not (self.index_照片的索引 is None):     self.otext_输出文本_当前样本序数.setText(str(self.index_照片的索引 + 1))
        if not (self.number_图片的物品数 is None):            self.otext_输出文本_实例总数.setText(str(self.number_图片的物品数))
        if not (self.img_标注区域数据用于显示 is None):
            self.label_标签_显示物品.setPixmap(QPixmap.fromImage(self.img_标注区域数据用于显示))
            # self.label_标签_显示物品.setScaledContents(True)

        if not (self.img_照片用于显示 is None):
            self.label_标签_显示图片.setPixmap(QPixmap.fromImage(self.img_照片用于显示))
            self.label_标签_显示图片.setScaledContents(True)
        self.otext_输出文本_物品序号.setText(str(self.index_照片中物品的索引 + 1))

        # ui状态及时设置
        # 数据集设置

        if self.radioButton_单选按钮_是否从文件夹导入并创建数据集.isChecked():
            self.itext_输入文本_导入图片的文件夹路径.setDisabled(0)
            self.itext_输入文本_fiftyone名称.setDisabled(0)
            self.itext_输入文本_路径_数据集标签文件目录.setDisabled(0)
            self.pushButton_按钮_导入并创建数据集.setDisabled(0)
            self.pushButton_按钮_选择标签文件.setDisabled(0)
            self.pushButton_按钮_选择图片目录.setDisabled(0)

        else:
            self.itext_输入文本_路径_数据集标签文件目录.setDisabled(1)
            self.itext_输入文本_导入图片的文件夹路径.setDisabled(1)
            self.itext_输入文本_fiftyone名称.setDisabled(1)
            self.pushButton_按钮_导入并创建数据集.setDisabled(1)
            self.pushButton_按钮_选择标签文件.setDisabled(1)
            self.pushButton_按钮_选择图片目录.setDisabled(1)

        # 图片样本设置
        if self.index_照片的索引 is not None and self.number_总样本数 is not None:
            self.pushButton_按钮_保存图片.setDisabled(0)
            self.radioButton_单选按钮_显示旋转框.setDisabled(0)
            self.radioButton_单选按钮_显示分割.setDisabled(0)
            self.radioButton_单选按钮_显示正框.setDisabled(0)
            self.pushButton_按钮_删除照片.setDisabled(0)
            self.itext_输入文本_图片样本路径.setDisabled(0)

            if self.index_照片的索引 == 0:
                self.pushButton_按钮_上一图片.setDisabled(1)
                self.pushButton_按钮_下一图片.setDisabled(0)
            elif self.index_照片的索引 == self.number_总样本数 - 1:
                self.pushButton_按钮_上一图片.setDisabled(0)
                self.pushButton_按钮_下一图片.setDisabled(1)
            else:
                self.pushButton_按钮_上一图片.setDisabled(0)
                self.pushButton_按钮_下一图片.setDisabled(0)
        else:
            self.pushButton_按钮_上一图片.setDisabled(1)
            self.pushButton_按钮_下一图片.setDisabled(1)
            self.pushButton_按钮_保存图片.setDisabled(1)
            self.pushButton_按钮_删除照片.setDisabled(1)
            self.radioButton_单选按钮_显示旋转框.setDisabled(1)
            self.radioButton_单选按钮_显示分割.setDisabled(1)
            self.radioButton_单选按钮_显示正框.setDisabled(1)
            self.itext_输入文本_图片样本路径.setDisabled(1)

        if self.point_新实例的起始点.__len__() == 2 and self.point_新实例的终点.__len__() == 2:
            self.pushButton_按钮_保存新实例.setDisabled(0)
        else:
            self.pushButton_按钮_保存新实例.setDisabled(1)

        # ui标注区域设置
        if self.label_区域标注体 is None:
            self.pushButton_按钮_上一物品.setDisabled(1)
            self.pushButton_按钮_下一物品.setDisabled(1)
            self.pushButton_按钮_标注物品.setDisabled(1)
            self.pushButton_按钮_标注背景.setDisabled(1)
            self.pushButton_按钮_重新标注.setDisabled(1)
            self.pushButton_按钮_保存物品.setDisabled(1)
            self.slider_画笔尺寸滑块.setDisabled(1)
            self.pushButton_按钮_删除实例.setDisabled(1)
        else:
            self.pushButton_按钮_标注物品.setDisabled(0)
            self.pushButton_按钮_标注背景.setDisabled(0)
            self.pushButton_按钮_重新标注.setDisabled(0)
            self.slider_画笔尺寸滑块.setDisabled(0)
            self.pushButton_按钮_删除实例.setDisabled(0)

            if (self.label_区域标注体.mask_物品bbox蒙版 is None or
                    self.label_区域标注体.bbox_包围框 is None or
                    self.label_区域标注体.box_最小矩形框 is None or
                    self.text_标签1 is None):
                self.pushButton_按钮_保存物品.setDisabled(1)
            else:
                self.pushButton_按钮_保存物品.setDisabled(0)

            if self.index_照片中物品的索引 == self.number_图片的物品数 - 1:
                self.pushButton_按钮_上一物品.setDisabled(0)
                self.pushButton_按钮_下一物品.setDisabled(1)
            elif self.index_照片中物品的索引 == 0:
                self.pushButton_按钮_上一物品.setDisabled(1)
                self.pushButton_按钮_下一物品.setDisabled(0)
            else:
                self.pushButton_按钮_上一物品.setDisabled(0)
                self.pushButton_按钮_下一物品.setDisabled(0)

        # 更新输入状态

        self.text_标签1 = self.itext_输入文本_标签1.toPlainText()
        self.text_标签2 = self.itext_输入文本_标签2.toPlainText()
        self.text_标签3 = self.itext_输入文本_标签3.toPlainText()
        # 快捷键设置

        # k = cv.waitKey(30) & 0xFF
        # if  self.label_区域标注体 is not None:
        #     if k == ord('1'):
        #         self.label_区域标注体.draw_绘制状态 = enum_绘制状态.绘制前景
        #     elif k == ord('2'):
        #         self.label_区域标注体.draw_绘制状态 = enum_绘制状态.绘制背景
        #     elif k == ord("3"):
        #         self.label_区域标注体.refresh_重新绘制()

    #############

    # 数据集相关
    #############
    def select_dir_选择图片目录(self):
        file_dialog = QFileDialog()
        pth_文件夹路径 = file_dialog.getExistingDirectory(self, "选择图片所在目录")
        # Export the dataset
        if pth_文件夹路径 is not '':
            self.itext_输入文本_导入图片的文件夹路径.setText(pth_文件夹路径)

    def select_file_选择标签文件(self):
        file_dialog = QFileDialog()
        # save_file_path = file_dialog.getSaveFileName(self, "Dump content", "", "HTML (*.html);; CSV (*.csv);;TXT (*.txt);;JSON (*.json);;SQL (*.Sql)")
        pth_文件目录 = file_dialog.getExistingDirectory(self, "选择标签文件目录")
        # Export the dataset
        if pth_文件目录 is not '':
            self.itext_输入文本_路径_数据集标签文件目录.setText(pth_文件目录)

    def create_创建数据集(self):
        if self.radioButton_单选按钮_是否从文件夹导入并创建数据集.isChecked():
            pth_image_dir = self.itext_输入文本_导入图片的文件夹路径.toPlainText()
            dataset_name = self.itext_输入文本_fiftyone名称.toPlainText()
            pth_标签文件路径 = self.itext_输入文本_路径_数据集标签文件目录.toPlainText()
            if pth_image_dir is "":
                self.feedback_软件提示("创建失败，请选着图片数据的目录")
                return
            if dataset_name in fo.list_datasets():
                self.feedback_软件提示("创建失败,已存在数据集：" + dataset_name + ",请更换名称")
                return
            if dataset_name is "":
                self.feedback_软件提示("创建失败,请输入要创建的数据集名称")
                return
            if pth_标签文件路径 == "":
                dataset_新数据集 = fo.Dataset.from_images_dir(images_dir=pth_image_dir)
                dataset_新数据集.name = dataset_name
                dataset_新数据集.persistent = True
            else:
                type_dataset = fo.types.FiftyOneDataset
                dataset_新数据集 = fo.Dataset.from_dir(dataset_dir=pth_标签文件路径,
                                                   dataset_type=type_dataset)
                dataset_新数据集.name = dataset_name
                for sample in dataset_新数据集:
                    sample.filepath = sample.filepath.replace(os.path.dirname(sample.filepath), pth_image_dir)
                    sample.save()
                dataset_新数据集.save()
                dataset_新数据集.persistent = True
            self.comboBox_下拉选项_数据集列表.addItem(dataset_name)
            self.comboBox_下拉选项_数据集列表.setCurrentText(dataset_name)
            self.feedback_软件提示("已成功创建数据集：" + dataset_name)

    def delete_删除数据集(self):
        dataset_name = self.comboBox_下拉选项_数据集列表.currentText()
        self.comboBox_下拉选项_数据集列表.removeItem(self.comboBox_下拉选项_数据集列表.currentIndex())
        dataset = fo.load_dataset(dataset_name)
        dataset.delete()
        self.feedback_软件提示(dataset_name + "数据集被成功删除")

    def update_刷新数据集(self):
        self.ini_初始化界面()

    def load_载入FiftyOne数据集(self):
        self.name_dataset_数据集名称 = self.comboBox_下拉选项_数据集列表.currentText()
        flag_检查结果, info_提示信息 = self.check_检查规范(self.name_dataset_数据集名称, type_类型=enum_检查项.数据库名称)
        if enum_检查结果.失败 == flag_检查结果: return
        self.dataset_fiftyone数据集 = fo.load_dataset(self.name_dataset_数据集名称)
        self.dataset_目标数据集 = self.dataset_fiftyone数据集
        self.Flag_是否已成功载入数据集 = True
        self.analyze_分析数据集(flag_是否重新载入数据集=True)
        # 图片
        self.clear_图片信息清空()
        self.read_载入图片样本()
        self.refresh_图片标注初始化()
        self.drawing_绘制原始图片()
        # 实例
        self.load_载入标注区域()
        self.drawing_绘制标注区域图片()

    def load_载入部分数据集(self):

        name_class_类别 = self.comboBox_类别过滤.currentText()
        match = (F("label"))
        if name_class_类别 == "all":
            self.dataset_目标数据集 = self.dataset_fiftyone数据集
        else:
            match_dataset_view = self.dataset_fiftyone数据集.filter_labels(field="ground_truth",
                                                                        filter=(match == name_class_类别))
            self.dataset_目标数据集 = match_dataset_view.clone()
        # 图片
        self.analyze_分析数据集()
        self.clear_图片信息清空()
        self.read_载入图片样本()
        self.refresh_图片标注初始化()
        self.drawing_绘制原始图片()
        # 实例
        self.load_载入标注区域()
        self.drawing_绘制标注区域图片()

    def analyze_分析数据集(self, flag_是否重新载入数据集=False):
        self.number_总样本数 = self.dataset_目标数据集.__len__()
        if flag_是否重新载入数据集:
            self.class_类别列表 = []
            self.comboBox_类别过滤.clear()
            self.comboBox_类别过滤.addItem("all")
            if self.dataset_fiftyone数据集.has_sample_field("ground_truth"):
                for sample in self.dataset_fiftyone数据集.select_fields("ground_truth"):
                    flag, _ = self.check_判断是否存在相应的字段(sample=sample, field_名称="detections")
                    if flag:
                        for detection in sample.ground_truth.detections:
                            label_标签 = detection.label
                            if not (label_标签 in self.class_类别列表):
                                self.class_类别列表.append(label_标签)
                                self.comboBox_类别过滤.addItem(label_标签)
        '''
        已标注数
        已标注数中的1例数到13例数
        '''
        self.index_照片的索引 = 0

    def export_导出数据集(self):
        file_dialog = QFileDialog()
        # save_file_path = file_dialog.getSaveFileName(self, "Dump content", "", "HTML (*.html);; CSV (*.csv);;TXT (*.txt);;JSON (*.json);;SQL (*.Sql)")
        data_type = fo.types.FiftyOneDataset

        dataset_name = self.comboBox_下拉选项_数据集列表.currentText()

        pth_保存路径 = file_dialog.getExistingDirectory(self, "导出数据集的文件夹"
                                                    )
        label_field = "ground_truth"  # for example
        # Export the dataset
        dataset_准备导出的数据 = fo.load_dataset(dataset_name)
        if pth_保存路径 is not '':
            self.feedback_软件提示("正在导出数据集，请耐心等待......" + pth_保存路径)
            dataset_准备导出的数据.export(export_dir=pth_保存路径, export_media=False, dataset_type=data_type,
                                   label_field=label_field, num_workers=1)

            self.feedback_软件提示("数据集已导出成功，导出路径为为:" + pth_保存路径)
        else:
            self.feedback_软件提示("数据集导出被取消")

    def display_网页端显示(self):
        session = fo.launch_app(self.dataset_fiftyone数据集)
        session.wait()

    def read_载入指定图片(self):
        pth_image = self.itext_输入文本_图片样本路径.toPlainText()
        if pth_image is "":
            self.feedback_软件提示("请填写当前图片路径")
        else:
            if os.path.exists(pth_image):
                self.sample_照片样本 = self.dataset_fiftyone数据集[pth_image]
            else:
                self.feedback_软件提示("当前图片不存在")
        self.read_载入图片样本(flag_指定图片=True)
        self.drawing_绘制原始图片()
        self.load_载入标注区域()
        self.drawing_绘制标注区域图片()

    def read_载入图片样本(self, flag_指定图片=False):
        if flag_指定图片:
            pass
        else:
            if self.dataset_目标数据集.__len__() == 0:
                self.feedback_软件提示("当前数据集样本为零")
                return
            if self.index_照片的索引 < self.number_总样本数:
                sample_照片样本 = self.dataset_目标数据集.skip(self.index_照片的索引).limit(1).first()
                self.sample_照片样本 = self.dataset_fiftyone数据集[sample_照片样本.filepath]
        self.pth_当前图片路径 = self.sample_照片样本.filepath
        self.itext_输入文本_图片样本路径.setText(str(self.pth_当前图片路径))
        result_检查结果, info_提示信息 = self.check_检查规范(self.pth_当前图片路径, type_类型=enum_检查项.图片路径)
        if enum_检查结果.失败 == result_检查结果: return
        self.img_照片原始数据 = cv.imread(self.pth_当前图片路径)
        self.height_图片样本高, self.width_图片样本宽 = self.img_照片原始数据.shape[0:2]
        self.number_图片的物品数 = 0
        flag_检测, value_检测 = self.check_判断是否存在相应的字段(self.sample_照片样本, "detections")
        if flag_检测:
            self.number_图片的物品数 = value_检测.__len__()
        else:
            self.number_图片的物品数 = 0
        self.index_照片中物品的索引 = 0
        self.point_新实例的终点 = ()
        self.point_新实例的起始点 = ()

    def drawing_绘制原始图片(self):
        img_目标照片 = self.img_照片原始数据.copy()
        self.Flag_显示正框 = self.radioButton_单选按钮_显示正框.isChecked()
        self.Flag_显示旋转框 = self.radioButton_单选按钮_显示旋转框.isChecked()
        self.Flag_显示分割结果 = self.radioButton_单选按钮_显示分割.isChecked()
        flag_有实例, _ = self.check_判断是否存在相应的字段(self.sample_照片样本, "detections")
        if flag_有实例:
            for detection_单个物品 in self.sample_照片样本.ground_truth.detections:
                if detection_单个物品["bounding_box"] is not None:
                    x_nor, y_nor, w_nor, h_nor = detection_单个物品["bounding_box"][:]
                    x = np.floor(x_nor * self.width_图片样本宽).__int__()
                    y = np.floor(y_nor * self.height_图片样本高).__int__()
                    w = np.floor(w_nor * self.width_图片样本宽).__int__()
                    h = np.floor(h_nor * self.height_图片样本高).__int__()
                    if self.Flag_显示正框:
                        color_正框 = self.color_正框.value

                        thickness_正框 = min(int(self.thickness_scan_正框画笔比例 * max(self.height_图片样本高, self.width_图片样本宽)),
                                           10)
                        thickness_正框 = max(thickness_正框, 1)
                        img_目标照片 = cv.rectangle(img_目标照片, (x, y), (x + w, y + h), color_正框, thickness=thickness_正框)
                if self.Flag_显示旋转框 and detection_单个物品.has_attribute("rbox") and detection_单个物品["rbox"] is not None:
                    x_rbox, y_rbox, w_rbox, h_rbox, a_rbox = detection_单个物品["rbox"]
                    rbox_cv_格式 = ((x_rbox, y_rbox), (w_rbox, h_rbox), a_rbox)
                    rbox_4点 = np.int0(cv.boxPoints(rbox_cv_格式))
                    thickness_旋转框 = min(int(self.thickness_scan_旋转框画笔比例 * max(self.height_图片样本高, self.width_图片样本宽)),
                                        10)
                    thickness_旋转框 = max(thickness_旋转框, 1)
                    img_目标照片 = RotBoxLabel.drawing_绘制矩形框(img_目标图片=img_目标照片, rbox_旋转=rbox_4点, thickness=thickness_旋转框)
                if self.Flag_显示分割结果 and detection_单个物品["mask"] is not None and detection_单个物品[
                    "bounding_box"] is not None:
                    mask_物品蒙版 = detection_单个物品["mask"]
                    indexY_实例索引, indexX_实例索引 = np.where(mask_物品蒙版 == 1)
                    indexX_实例索引 += x
                    indexY_实例索引 += y
                    img_目标照片[:, :, 0][indexY_实例索引, indexX_实例索引] = 255

        if self.point_新实例的起始点.__len__() == 2 and self.point_新实例的终点.__len__() == 2:
            thickness_正框 = min(int(self.thickness_scan_正框画笔比例 * max(self.height_图片样本高, self.width_图片样本宽)),
                               10)
            thickness_正框 = max(thickness_正框, 1)
            img_目标照片 = cv.rectangle(img_目标照片, self.point_新实例的起始点, self.point_新实例的终点, enum_颜色.红色.value, thickness_正框)

        #############################################
        # img_目标照片 改变下尺寸
        img_目标照片 = cv.cvtColor(img_目标照片, cv.COLOR_BGR2RGB)
        size_较长的尺寸 = max(self.width_图片样本宽, self.height_图片样本高)
        self.scan_图片缩放比 = self.size_图片显示区域尺寸 / size_较长的尺寸
        self.width_图片缩放后的宽 = int(self.scan_图片缩放比 * self.width_图片样本宽)
        self.height_图片缩放后的高 = int(self.scan_图片缩放比 * self.height_图片样本高)
        self.label_标签_显示图片.resize(self.width_图片缩放后的宽, self.height_图片缩放后的高)
        img_目标照片 = cv.resize(img_目标照片, (self.width_图片缩放后的宽, self.height_图片缩放后的高))
        if len(img_目标照片.shape) == 3:
            img_目标照片 = QImage(img_目标照片, img_目标照片.shape[1], img_目标照片.shape[0], img_目标照片.strides[0],
                              QImage.Format_RGB888)
        elif len(img_目标照片.shape) == 2:
            img_目标照片 = QImage(img_目标照片, img_目标照片.shape[1], img_目标照片.shape[0], img_目标照片.strides[0],
                              QImage.Format_Grayscale8)
        self.img_照片用于显示 = img_目标照片

    def next_下一张图片(self):
        if self.Flag_是否已成功载入数据集:
            if self.index_照片的索引 < self.number_总样本数 - 1:
                self.index_照片的索引 += 1
                self.pushButton_按钮_上一图片.setDisabled(0)
            if self.index_照片的索引 == self.number_总样本数 - 1:
                self.pushButton_按钮_下一图片.setDisabled(1)
            self.read_载入图片样本()
            self.drawing_绘制原始图片()
            self.load_载入标注区域()
            self.drawing_绘制标注区域图片()
        else:
            self.feedback_软件提示(str_提示内容=enum_软件状态.没有数据集.value, flag_是否警告=True)

    def front_上一张图片(self):
        if self.Flag_是否已成功载入数据集:
            if 0 < self.index_照片的索引 < self.number_总样本数:
                self.index_照片的索引 -= 1
                self.pushButton_按钮_下一图片.setDisabled(0)
            if 0 == self.index_照片的索引:
                self.pushButton_按钮_上一图片.setDisabled(1)
            self.read_载入图片样本()
            self.drawing_绘制原始图片()
            self.load_载入标注区域()
            self.drawing_绘制标注区域图片()
        else:
            self.feedback_软件提示(str_提示内容=enum_软件状态.没有数据集.value, flag_是否警告=True)

    def save_保存图片(self, flag_下一张=True):
        self.sample_照片样本.save()
        self.dataset_目标数据集.save()
        if flag_下一张:
            self.next_下一张图片()
        self.feedback_软件提示("已保存图片")

    def delete_删除这张照片(self):
        self.dataset_目标数据集.delete_samples(self.sample_照片样本)
        self.dataset_目标数据集.save()
        self.number_总样本数 = self.dataset_目标数据集.__len__()
        self.read_载入图片样本()
        self.drawing_绘制原始图片()
        self.load_载入标注区域()
        self.drawing_绘制标注区域图片()

    def clear_图片信息清空(self):
        self.number_图片的物品数 = 0
        self.index_照片的索引 = 0

    def crate_新建实例(self):
        self.Flag_是否新建实例 = True
        pass

    # 重新标注图片
    def refresh_图片标注初始化(self):
        self.clear_清理标注区相关信息()
        if self.number_图片的物品数 != 0:
            self.index_照片中物品的索引 = 0
            self.detection_当前物品 = self.sample_照片样本.ground_truth.detections[self.index_照片中物品的索引]
            self.bbox_待标注的区域 = self.detection_当前物品["bounding_box"]
            self.text_标签1 = self.detection_当前物品['label']

    @staticmethod
    def check_判断是否存在相应的字段(sample: fo.Sample, field_名称: str):
        flag_是否存在 = False
        value_值 = None
        if "ground_truth" == field_名称:
            flag_是否存在 = sample.has_field("ground_truth")
            if flag_是否存在:
                value_值 = sample[field_名称]
        elif "detections" == field_名称:
            if sample.has_field("ground_truth"):
                if sample["ground_truth"] is not None:
                    flag_是否存在 = sample["ground_truth"].has_field("detections")
                    if flag_是否存在:
                        value_值 = sample["ground_truth"]["detections"]

        elif "bounding_box" == field_名称:
            if sample.has_field("ground_truth"):
                if sample["ground_truth"] is not None:
                    if sample["ground_truth"].has_field("detections"):
                        flag_是否存在 = True
                        if sample["ground_truth"]["detections"].__len__() != 0:
                            value_值 = sample["ground_truth"]["detections"][0]["bounding_box"]
        return flag_是否存在, value_值

    #############
    # 标注区域相关
    #############
    def load_载入标注区域(self):
        self.clear_清理标注区相关信息()
        flag_检测, value_检测 = self.check_判断是否存在相应的字段(self.sample_照片样本, "detections")
        if flag_检测:
            self.number_图片的物品数 = value_检测.__len__()
        else:
            self.number_图片的物品数 = 0
        flag_是否有检测结果, value_bbox = self.check_判断是否存在相应的字段(self.sample_照片样本, "bounding_box")
        if flag_是否有检测结果 and value_bbox is not None:
            if value_bbox.__len__() == 4:
                self.detection_当前物品 = self.sample_照片样本.ground_truth.detections[self.index_照片中物品的索引]
        else:
            self.index_照片中物品的索引 = 0
            self.detection_当前物品 = fo.Detection()
            self.bbox_待标注的区域 = [0, 0, self.width_图片样本宽, self.height_图片样本高]

        if self.detection_当前物品.label is not None and self.detection_当前物品.label is not "":
            if self.radioButton_单选按钮_锁定标签.isChecked():
                pass
            else:
                self.text_标签1 = self.detection_当前物品['label']
                self.itext_输入文本_标签1.setText(self.text_标签1)

        if self.detection_当前物品.bounding_box.__len__() == 4:
            x_nor, y_nor, w_nor, h_nor = self.detection_当前物品["bounding_box"]
            x = np.floor(x_nor * self.width_图片样本宽).__int__()
            y = np.floor(y_nor * self.height_图片样本高).__int__()
            w = np.floor(w_nor * self.width_图片样本宽).__int__()
            h = np.floor(h_nor * self.height_图片样本高).__int__()
            self.bbox_待标注的区域 = [x, y, w, h]

        if self.bbox_待标注的区域 is not None and self.bbox_待标注的区域.__len__() == 4:
            x_左上, y_左上, w_宽, h_高 = self.bbox_待标注的区域
            x_左上_new = x_左上 - 50
            y_左上_new = y_左上 - 50

            x_左上_new = max(x_左上_new, 0)
            y_左上_new = max(y_左上_new, 0)
            w_减少的 = x_左上 - x_左上_new
            h_减少的 = y_左上 - y_左上_new
            x_左上 = x_左上_new
            y_左上 = y_左上_new
            w_宽 += 50 + w_减少的
            h_高 += 50 + h_减少的
            w_宽 = min(self.width_图片样本宽 - x_左上, w_宽)
            h_高 = min(self.height_图片样本高 - y_左上, h_高)
            self.bbox_待标注的区域 = [x_左上, y_左上, w_宽, h_高]
            self.height_标注区域高 = h_高
            self.width_标注区域宽 = w_宽
            img_待标注的区域 = self.img_照片原始数据[y_左上:y_左上 + h_高, x_左上:x_左上 + w_宽, :]

            self.size_画笔尺寸 = self.slider_画笔尺寸滑块.value()
            self.label_区域标注体 = RotBoxLabel(img_目标图像=img_待标注的区域, size_画笔尺寸=self.size_画笔尺寸)

            if self.detection_当前物品.bounding_box.__len__() == 4:
                x_bbox, y_bbox, w_bbox, h_bbox = self.detection_当前物品.bounding_box
                x_bbox = int(x_bbox * self.width_图片样本宽)
                y_bbox = int(y_bbox * self.height_图片样本高)
                w_bbox = int(w_bbox * self.width_图片样本宽)
                h_bbox = int(h_bbox * self.height_图片样本高)
                x_bbox = x_bbox - self.bbox_待标注的区域[0]
                y_bbox = y_bbox - self.bbox_待标注的区域[1]
                self.label_区域标注体.bbox_包围框 = [x_bbox, y_bbox, w_bbox, h_bbox]

            if self.detection_当前物品.has_field("rbox"):
                if self.detection_当前物品["rbox"].__len__() == 5:
                    x_rbox, y_rbox, w_rbox, h_rbox, a_rbox = self.detection_当前物品["rbox"]
                    x_rbox = x_rbox - self.bbox_待标注的区域[0]
                    y_rbox = y_rbox - self.bbox_待标注的区域[1]
                    self.label_区域标注体.box_最小矩形框 = ((x_rbox, y_rbox), (w_rbox, h_rbox), a_rbox)
                    self.label_区域标注体.drawing_初始化绘制()

            if self.detection_当前物品.mask is not None:
                self.label_区域标注体.mask_物品bbox蒙版 = self.detection_当前物品.mask
                self.label_区域标注体.drawing_初始化绘制()

            self.drawing_绘制标注区域图片()

    def keyPressEvent(self, event):
        if event.key() == ord("1"):
            if enum_控件对象.图片区域 == self.mouse_press_object_鼠标单击对象:
                self.front_上一张图片()
            elif enum_控件对象.标注区域 == self.mouse_press_object_鼠标单击对象:
                if self.number_图片的物品数 <= 1 or self.index_照片中物品的索引 == 0:
                    self.front_上一张图片()
                else:
                    self.front_上一个物品()
        elif ord("2") == event.key():
            if enum_控件对象.图片区域 == self.mouse_press_object_鼠标单击对象:
                self.next_下一张图片()
            elif enum_控件对象.标注区域 == self.mouse_press_object_鼠标单击对象:
                if self.number_图片的物品数 <= 1 or self.index_照片中物品的索引 == self.number_图片的物品数 - 1:
                    self.next_下一张图片()
                else:
                    self.next_下一个物品()
        elif ord("3") == event.key():
            self.refresh_重新标注物品()
        elif ord('4') == event.key():
            if enum_控件对象.图片区域 == self.mouse_press_object_鼠标单击对象:
                if self.point_新实例的起始点.__len__() == 2 and self.point_新实例的终点.__len__() == 2:
                    self.save_保存新建的物品()
                else:
                    self.save_保存图片()
            elif enum_控件对象.标注区域 == self.mouse_press_object_鼠标单击对象:
                self.save_object_保存物品()
        elif ord("5") == event.key():
            if enum_控件对象.图片区域 == self.mouse_press_object_鼠标单击对象:
                self.delete_删除这张照片()
            elif enum_控件对象.标注区域 == self.mouse_press_object_鼠标单击对象:
                if self.number_图片的物品数 == 0:
                    self.delete_删除这张照片()
                else:
                    self.delete_删除当前实例()

    def label_标注背景区域(self):
        if not (self.label_区域标注体 is None):
            self.label_区域标注体.draw_绘制状态 = enum_绘制状态.绘制背景
            self.feedback_软件提示("请标注背景区域")

    def label_标注物品区域(self):
        if not (self.label_区域标注体 is None):
            self.label_区域标注体.draw_绘制状态 = enum_绘制状态.绘制前景
            self.feedback_软件提示("请标注前景区域")

    def drawing_绘制标注区域图片(self):
        if not (self.label_区域标注体 is None):
            img_物品交互照片 = self.label_区域标注体.img_交互显示图像.copy()
            # img_目标照片 改变下尺寸
            img_物品交互照片 = cv.cvtColor(img_物品交互照片, cv.COLOR_BGR2RGB)
            size_较长的尺寸 = max(self.width_标注区域宽, self.height_标注区域高)
            self.scan_标注区域缩放比 = self.area_size_区域尺寸 / size_较长的尺寸
            self.width_物品缩放后的宽 = int(self.scan_标注区域缩放比 * self.width_标注区域宽)
            self.height_物品缩放后的高 = int(self.scan_标注区域缩放比 * self.height_标注区域高)
            self.label_标签_显示物品.resize(self.width_物品缩放后的宽, self.height_物品缩放后的高)
            img_物品交互照片 = cv.resize(img_物品交互照片, (self.width_物品缩放后的宽, self.height_物品缩放后的高))
            img_物品交互照片 = QImage(img_物品交互照片, img_物品交互照片.shape[1], img_物品交互照片.shape[0], img_物品交互照片.strides[0],
                                QImage.Format_RGB888)
            self.img_标注区域数据用于显示 = img_物品交互照片
        else:
            self.img_标注区域数据用于显示 = None

    def refresh_重新标注物品(self):
        self.label_区域标注体.refresh_重新绘制()
        self.drawing_绘制标注区域图片()

    def next_下一个物品(self):
        if 0 <= self.index_照片中物品的索引 < self.number_图片的物品数 - 1:
            self.index_照片中物品的索引 += 1
            self.pushButton_按钮_上一物品.setDisabled(0)
        if self.index_照片中物品的索引 == self.number_图片的物品数 - 1:
            self.pushButton_按钮_下一物品.setDisabled(1)
        self.load_载入标注区域()

    def front_上一个物品(self):
        if 0 < self.index_照片中物品的索引 <= self.number_图片的物品数 - 1:
            self.index_照片中物品的索引 -= 1
            self.pushButton_按钮_下一物品.setDisabled(0)
        if self.index_照片中物品的索引 == 0:
            self.pushButton_按钮_上一物品.setDisabled(1)
        self.load_载入标注区域()

    def save_object_保存物品(self):
        bbox_包围框 = self.label_区域标注体.bbox_包围框
        bbox_包围框 = list(bbox_包围框)
        bbox_包围框[0] += self.bbox_待标注的区域[0]
        bbox_包围框[1] += self.bbox_待标注的区域[1]
        bbox_包围框[0] = bbox_包围框[0] / self.width_图片样本宽
        bbox_包围框[1] = bbox_包围框[1] / self.height_图片样本高
        bbox_包围框[2] = bbox_包围框[2] / self.width_图片样本宽
        bbox_包围框[3] = bbox_包围框[3] / self.height_图片样本高

        self.detection_当前物品.bounding_box = bbox_包围框

        (x, y), (w, h), a = self.label_区域标注体.box_最小矩形框
        x += self.bbox_待标注的区域[0]
        y += self.bbox_待标注的区域[1]
        # x = x / self.width_图片样本宽
        # y = y / self.height_图片样本高
        # w = w / self.width_图片样本宽
        # h = h / self.height_图片样本高
        # a = a / 90.0
        self.detection_当前物品["rbox"] = [x, y, w, h, a]  # 重新定义的 rbox 字段
        # rbox_4_最小包围框 = self.label_区域标注体.box_最小包围框4点形式
        # for index, rbox_point in enumerate(rbox_4_最小包围框):
        #     rbox_4_最小包围框[index][0] += self.bbox_待标注的区域[0]
        #     rbox_4_最小包围框[index][1] += self.bbox_待标注的区域[1]
        # point1, point2, point3, point4 = rbox_4_最小包围框[:]
        # self.detection_当前物品["rbox_4p"] = [point1, point2, point3, point4]
        mask_object_物品实例掩膜 = self.label_区域标注体.mask_物品bbox蒙版
        self.detection_当前物品.mask = np.array(mask_object_物品实例掩膜)
        self.detection_当前物品.label = self.itext_输入文本_标签1.toPlainText()
        flag_detections, value_detections = self.check_判断是否存在相应的字段(self.sample_照片样本, "detections")
        if flag_detections and self.number_图片的物品数 != 0:
            self.sample_照片样本["ground_truth"]["detections"][self.index_照片中物品的索引] = self.detection_当前物品
        elif self.number_图片的物品数 == 0:
            self.sample_照片样本["ground_truth"] = fo.Detections(detections=[self.detection_当前物品])
        else:
            self.sample_照片样本["ground_truth"] = fo.Detections(detections=[self.detection_当前物品])

        if self.index_照片中物品的索引 == self.number_图片的物品数 - 1 or self.number_图片的物品数 == 0:
            self.save_保存图片()
        else:
            self.save_保存图片(flag_下一张=False)
            self.drawing_绘制原始图片()
            self.next_下一个物品()

    def save_保存新建的物品(self):
        if len(self.point_新实例的终点) == 2 and len(self.point_新实例的终点) == 2:
            x_point1, y_point1 = self.point_新实例的起始点
            x_point2, y_point2 = self.point_新实例的终点
            w = abs(x_point1 - x_point2) / self.width_图片样本宽
            h = abs(y_point1 - y_point2) / self.height_图片样本高
            x = min(x_point1, x_point2) / self.width_图片样本宽
            y = min(y_point1, y_point2) / self.height_图片样本高
            bbox_包围框 = [x, y, w, h]
            detection_新物品 = fo.Detection()
            detection_新物品.bounding_box = bbox_包围框
            flag_是否有检测标签, _ = self.check_判断是否存在相应的字段(self.sample_照片样本, "detections")
            if flag_是否有检测标签:
                self.sample_照片样本.ground_truth.detections.append(detection_新物品)
            else:
                self.sample_照片样本["ground_truth"] = fo.Detections(detections=[detection_新物品])
            self.point_新实例的终点 = ()
            self.point_新实例的起始点 = ()
            self.save_保存图片(flag_下一张=False)
            self.read_载入图片样本()
            self.drawing_绘制原始图片()
            self.index_照片中物品的索引 = self.number_图片的物品数 - 1
            self.load_载入标注区域()
            self.drawing_绘制标注区域图片()

    def delete_删除当前实例(self):
        if 0 <= self.index_照片中物品的索引 <= self.number_图片的物品数 - 1:
            del self.sample_照片样本["ground_truth"]["detections"][self.index_照片中物品的索引]
            self.save_保存图片(flag_下一张=False)
            self.read_载入图片样本()
            self.drawing_绘制原始图片()
            self.load_载入标注区域()
            self.drawing_绘制标注区域图片()

    def clear_清理标注区相关信息(self):
        self.width_物品缩放后的宽 = None
        self.height_物品缩放后的高 = None
        self.width_标注区域宽 = None
        self.height_标注区域高 = None
        self.label_区域标注体 = None

    #############
    # 其它
    #############
    def mousePressEvent(self, event):
        pos_位置 = event.pos()
        pos_图片群 = self.groupBox_图片显示区域群.mapFromParent(pos_位置)
        pos_图片位置 = self.label_标签_显示图片.mapFromParent(pos_图片群)
        x_图片, y_图片 = pos_图片位置.x(), pos_图片位置.y()
        if (0 < x_图片 < self.label_标签_显示图片.width()) and (0 < y_图片 < self.label_标签_显示图片.height()):
            self.mouse_press_object_鼠标单击对象 = enum_控件对象.图片区域
            x_原始图片位置 = int(np.floor(x_图片 / self.scan_图片缩放比))
            y_原始图片位置 = int(np.floor(y_图片 / self.scan_图片缩放比))
            self.point_新实例的起始点 = (x_原始图片位置, y_原始图片位置)
            self.point_新实例的终点 = ()

        pos_物品群 = self.groupBox_标注物品区域群.mapFromParent(pos_位置)
        pos_物品位置 = self.label_标签_显示物品.mapFromParent(pos_物品群)
        x_物品, y_物品 = pos_物品位置.x(), pos_物品位置.y()
        if (0 < x_物品 < self.label_标签_显示物品.width()) and (0 < y_物品 < self.label_标签_显示物品.height()):
            self.mouse_press_object_鼠标单击对象 = enum_控件对象.标注区域

    def mouseMoveEvent(self, event_标注物品):
        pos_位置 = event_标注物品.pos()
        pos_图片群 = self.groupBox_图片显示区域群.mapFromParent(pos_位置)
        pos_图片位置 = self.label_标签_显示图片.mapFromParent(pos_图片群)
        x_图片, y_图片 = pos_图片位置.x(), pos_图片位置.y()
        if (0 < x_图片 < self.label_标签_显示图片.width()) and (0 < y_图片 < self.label_标签_显示图片.height()):
            x_原始图片位置 = int(np.floor(x_图片 / self.scan_图片缩放比))
            y_原始图片位置 = int(np.floor(y_图片 / self.scan_图片缩放比))
            self.point_新实例的终点 = (x_原始图片位置, y_原始图片位置)
            self.drawing_绘制原始图片()
        pos_物品群 = self.groupBox_标注物品区域群.mapFromParent(pos_位置)
        pos_物品位置 = self.label_标签_显示物品.mapFromParent(pos_物品群)
        x_物品, y_物品 = pos_物品位置.x(), pos_物品位置.y()
        if (0 < x_物品 < self.label_标签_显示物品.width()) and (0 < y_物品 < self.label_标签_显示物品.height()):
            if not (self.label_区域标注体 is None):
                if event_标注物品.buttons() == QtCore.Qt.LeftButton:
                    self.label_区域标注体.draw_绘制状态 = enum_绘制状态.绘制前景
                elif event_标注物品.buttons() == QtCore.Qt.RightButton:
                    self.label_区域标注体.draw_绘制状态 = enum_绘制状态.绘制背景
                _event_opencv = cv.EVENT_MOUSEMOVE
                _flags = cv.EVENT_FLAG_LBUTTON
                x_原始物品位置 = int(np.floor(x_物品 / self.scan_标注区域缩放比))
                y_原始物品位置 = int(np.floor(y_物品 / self.scan_标注区域缩放比))
                print("x_原始物品位置", x_原始物品位置, y_原始物品位置)
                self.label_区域标注体.draw_circle(event=_event_opencv, x=x_原始物品位置, y=y_原始物品位置, flags=_flags, param=None)
                self.drawing_绘制标注区域图片()

    def mouseReleaseEvent(self, event) -> None:
        pos_位置 = event.pos()
        pos_图片群 = self.groupBox_图片显示区域群.mapFromParent(pos_位置)
        pos_图片位置 = self.label_标签_显示图片.mapFromParent(pos_图片群)
        x_图片, y_图片 = pos_图片位置.x(), pos_图片位置.y()
        if (0 < x_图片 < self.label_标签_显示图片.width()) and (0 < y_图片 < self.label_标签_显示图片.height()):
            # if self.radioButton_单选按钮_新建新实例.isChecked():
            x_原始图片位置 = int(np.floor(x_图片 / self.scan_图片缩放比))
            y_原始图片位置 = int(np.floor(y_图片 / self.scan_图片缩放比))
            self.point_新实例的终点 = (x_原始图片位置, y_原始图片位置)
        pos_物品群 = self.groupBox_标注物品区域群.mapFromParent(pos_位置)
        pos_位置 = self.label_标签_显示物品.mapFromParent(pos_物品群)
        x_物品, y_物品 = pos_位置.x(), pos_位置.y()
        if (0 < x_物品 < self.label_标签_显示物品.width()) and (0 < y_物品 < self.label_标签_显示物品.height()):
            if not (self.label_区域标注体 is None):
                print(x_物品, y_物品)
                _event_opencv = cv.EVENT_LBUTTONUP
                x_物品 = int(np.floor(x_物品 / self.scan_标注区域缩放比))
                y_物品 = int(np.floor(y_物品 / self.scan_标注区域缩放比))
                self.label_区域标注体.draw_circle(event=_event_opencv, x=x_物品, y=y_物品, flags=None, param=None)
                self.drawing_绘制标注区域图片()

    def change_更改画笔尺寸(self, value_值):
        self.size_画笔尺寸 = int(value_值)
        self.label_区域标注体.size_画笔大小 = int(self.size_画笔尺寸)

    def feedback_软件提示(self, str_提示内容: str, flag_是否警告: bool = False):
        str_提示内容 = "提示：" + str_提示内容
        if flag_是否警告:
            self.label_view_输出文本_提示信息.setText(str_提示内容)  # 红色
        else:
            self.label_view_输出文本_提示信息.setText(str_提示内容)  # 绿色

    def check_检查规范(self, item_项目, type_类型: enum_检查项):
        """
        检查输入的规范，输出检查结果与信息
        :param item_项目:
        :param type_类型:
        :return:
        """
        info_信息: enum_检查结果信息 = enum_检查结果信息.默认
        result_检查结果 = enum_检查结果.成功
        if enum_检查项.数据库名称 == type_类型:
            set_list_现有数据集 = fo.list_datasets()
            print(set_list_现有数据集)
            print("item_项目", item_项目)
            result_检查结果 = enum_检查结果.成功 if (item_项目 in set_list_现有数据集) else enum_检查结果.失败
            info_信息 = enum_检查结果信息.数据库不存在 if (enum_检查结果.失败 == result_检查结果) else enum_检查结果信息.默认
        # elif:
        #     print("其它项")

        if enum_检查结果.失败 == result_检查结果:
            self.feedback_软件提示(info_信息.value, flag_是否警告=True)
            self.Flag_是否已成功载入数据集 = False
        else:
            self.feedback_软件提示(info_信息.value)

        return result_检查结果, info_信息


if __name__ == '__main__':
    '''
    ls_现有数据库列表 = fo.list_datasets()
    for dataset_某个数据库 in ls_现有数据库列表:
        if "Messy" in dataset_某个数据库:
            dataset_目标数据库 = dataset_某个数据库
    dataset_数据库 = fo.load_dataset(dataset_目标数据库)
    dataset_view_数据库 = dataset_数据库.view()
    for pic_一张图片 in dataset_view_数据库:  # 取出一张图片
        pth_文件路径 = pic_一张图片.filepath
        img_图片 = cv.imread(pth_文件路径)
        w_原始图片宽 = img_图片.shape[1]
        h_原始图片高 = img_图片.shape[0]
        detections_实例列表 = pic_一张图片.ground_truth.detections
        for index_实例序号, ins_某个物品 in enumerate(detections_实例列表):
            img_图片_标准板 = np.ones((h_原始图片高, w_原始图片宽, 3), np.uint8)
            bbox_检测框 = ins_某个物品.bounding_box
            label_实例标签 = ins_某个物品.label
            x_左上角_nor, y_左上角_nor, w_宽度_nor, h_高度_nor = bbox_检测框[:]
            x_左上角 = math.floor(x_左上角_nor * 1920)
            w_宽度 = math.floor(w_宽度_nor * 1920)
            y_左上角 = math.floor(y_左上角_nor * 1080)
            h_高度 = math.floor(h_高度_nor * 1080)
            img_图片_标准板[y_左上角:y_左上角 + h_高度, x_左上角:x_左上角 + w_宽度] = img_图片[y_左上角:y_左上角 + h_高度, x_左上角:x_左上角 + w_宽度]
            box_求解框 = RotBoxLabel(img_目标图像=img_图片_标准板, index_实例序号=index_实例序号 + 1)
            box_求解框.running_运行求解()
            rbox_旋转包围框 = box_求解框.box_最小矩形框
            rbox_4点形式 = box_求解框.box_最小包围框4点形式
            mask_物品蒙版 = box_求解框.mask_实例最小区域模板
            
            print(1)
            break
            # cv.imshow('目标', img_图片_标准板)
            # cv.waitKey(0)
    '''
    # img = cv.imread('j:\\2.png')
    # box_求解框 = RotBoxLabel(img_目标图像=img)
    # # a = LabelInPut()
    # # a.running_开始标注()
    # box_求解框.running_运行求解()
    QImageReader.supportedImageFormats()
    app = QApplication(sys.argv)
    app.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))
    window = LabelIt()
    window.show()
    sys.exit(app.exec_())
