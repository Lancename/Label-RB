# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'labelIt.ui'
##
## Created by: Qt User Interface Compiler version 6.1.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *  # type: ignore
from PySide6.QtGui import *  # type: ignore
from PySide6.QtWidgets import *  # type: ignore


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1635, 916)
        MainWindow.setMouseTracking(False)
        MainWindow.setLayoutDirection(Qt.LeftToRight)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayoutWidget_7 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_7.setObjectName(u"horizontalLayoutWidget_7")
        self.horizontalLayoutWidget_7.setGeometry(QRect(840, 120, 551, 41))
        self.horizontalLayout_10 = QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.horizontalLayoutWidget_7)
        self.label.setObjectName(u"label")

        self.horizontalLayout_10.addWidget(self.label)

        self.horizontalSlider_brush_size = QSlider(self.horizontalLayoutWidget_7)
        self.horizontalSlider_brush_size.setObjectName(u"horizontalSlider_brush_size")
        self.horizontalSlider_brush_size.setMaximum(30)
        self.horizontalSlider_brush_size.setValue(1)
        self.horizontalSlider_brush_size.setOrientation(Qt.Horizontal)

        self.horizontalLayout_10.addWidget(self.horizontalSlider_brush_size)

        self.pushButton_save_object = QPushButton(self.centralwidget)
        self.pushButton_save_object.setObjectName(u"pushButton_save_object")
        self.pushButton_save_object.setGeometry(QRect(1130, 690, 191, 51))
        self.label_12 = QLabel(self.centralwidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(10, 820, 60, 21))
        self.label_13 = QLabel(self.centralwidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(600, 850, 541, 20))
        self.label_state_feedback = QLabel(self.centralwidget)
        self.label_state_feedback.setObjectName(u"label_state_feedback")
        self.label_state_feedback.setGeometry(QRect(10, 850, 561, 16))
        self.progressBar_label = QProgressBar(self.centralwidget)
        self.progressBar_label.setObjectName(u"progressBar_label")
        self.progressBar_label.setGeometry(QRect(80, 820, 1461, 23))
        self.progressBar_label.setValue(24)
        self.label_label_number_process = QLabel(self.centralwidget)
        self.label_label_number_process.setObjectName(u"label_label_number_process")
        self.label_label_number_process.setGeometry(QRect(1550, 820, 66, 21))
        self.pushButton_save_picture = QPushButton(self.centralwidget)
        self.pushButton_save_picture.setObjectName(u"pushButton_save_picture")
        self.pushButton_save_picture.setGeometry(QRect(460, 700, 191, 41))
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(970, 750, 311, 16))
        self.splitter_8 = QSplitter(self.centralwidget)
        self.splitter_8.setObjectName(u"splitter_8")
        self.splitter_8.setGeometry(QRect(600, 780, 421, 31))
        self.splitter_8.setOrientation(Qt.Horizontal)
        self.label_number_of_1_object_2 = QLabel(self.splitter_8)
        self.label_number_of_1_object_2.setObjectName(u"label_number_of_1_object_2")
        self.splitter_8.addWidget(self.label_number_of_1_object_2)
        self.textBrowser_num_2_5_object = QTextBrowser(self.splitter_8)
        self.textBrowser_num_2_5_object.setObjectName(u"textBrowser_num_2_5_object")
        self.splitter_8.addWidget(self.textBrowser_num_2_5_object)
        self.label_number_of_1_object_3 = QLabel(self.splitter_8)
        self.label_number_of_1_object_3.setObjectName(u"label_number_of_1_object_3")
        self.splitter_8.addWidget(self.label_number_of_1_object_3)
        self.textBrowser_num_6_9_object = QTextBrowser(self.splitter_8)
        self.textBrowser_num_6_9_object.setObjectName(u"textBrowser_num_6_9_object")
        self.splitter_8.addWidget(self.textBrowser_num_6_9_object)
        self.label_number_of_1_object_4 = QLabel(self.splitter_8)
        self.label_number_of_1_object_4.setObjectName(u"label_number_of_1_object_4")
        self.splitter_8.addWidget(self.label_number_of_1_object_4)
        self.textBrowser_num_10_13_object = QTextBrowser(self.splitter_8)
        self.textBrowser_num_10_13_object.setObjectName(u"textBrowser_num_10_13_object")
        self.splitter_8.addWidget(self.textBrowser_num_10_13_object)
        self.splitter_9 = QSplitter(self.centralwidget)
        self.splitter_9.setObjectName(u"splitter_9")
        self.splitter_9.setGeometry(QRect(140, 780, 451, 31))
        self.splitter_9.setOrientation(Qt.Horizontal)
        self.splitter_6 = QSplitter(self.splitter_9)
        self.splitter_6.setObjectName(u"splitter_6")
        self.splitter_6.setOrientation(Qt.Horizontal)
        self.label_number_of_samples = QLabel(self.splitter_6)
        self.label_number_of_samples.setObjectName(u"label_number_of_samples")
        self.splitter_6.addWidget(self.label_number_of_samples)
        self.textBrowser_sum_images = QTextBrowser(self.splitter_6)
        self.textBrowser_sum_images.setObjectName(u"textBrowser_sum_images")
        self.splitter_6.addWidget(self.textBrowser_sum_images)
        self.splitter_9.addWidget(self.splitter_6)
        self.splitter_7 = QSplitter(self.splitter_9)
        self.splitter_7.setObjectName(u"splitter_7")
        self.splitter_7.setOrientation(Qt.Horizontal)
        self.label_number_of_label_samples = QLabel(self.splitter_7)
        self.label_number_of_label_samples.setObjectName(u"label_number_of_label_samples")
        self.splitter_7.addWidget(self.label_number_of_label_samples)
        self.textBrowser_num_labeled = QTextBrowser(self.splitter_7)
        self.textBrowser_num_labeled.setObjectName(u"textBrowser_num_labeled")
        self.splitter_7.addWidget(self.textBrowser_num_labeled)
        self.splitter_9.addWidget(self.splitter_7)
        self.label_number_of_1_object = QLabel(self.splitter_9)
        self.label_number_of_1_object.setObjectName(u"label_number_of_1_object")
        self.splitter_9.addWidget(self.label_number_of_1_object)
        self.textBrowser_num_1_object = QTextBrowser(self.splitter_9)
        self.textBrowser_num_1_object.setObjectName(u"textBrowser_num_1_object")
        self.splitter_9.addWidget(self.textBrowser_num_1_object)
        self.groupBox_label_object = QGroupBox(self.centralwidget)
        self.groupBox_label_object.setObjectName(u"groupBox_label_object")
        self.groupBox_label_object.setGeometry(QRect(840, 169, 550, 471))
        font = QFont()
        font.setPointSize(12)
        self.groupBox_label_object.setFont(font)
        self.groupBox_label_object.setAlignment(Qt.AlignCenter)
        self.label_image_object = QLabel(self.groupBox_label_object)
        self.label_image_object.setObjectName(u"label_image_object")
        self.label_image_object.setEnabled(True)
        self.label_image_object.setGeometry(QRect(70, 50, 400, 400))
        self.label_image_object.setMouseTracking(True)
        self.label_image_object.setAlignment(Qt.AlignCenter)
        self.pushButton_brush_background = QPushButton(self.groupBox_label_object)
        self.pushButton_brush_background.setObjectName(u"pushButton_brush_background")
        self.pushButton_brush_background.setEnabled(True)
        self.pushButton_brush_background.setGeometry(QRect(450, 10, 101, 31))
        self.pushButton_brush_object = QPushButton(self.groupBox_label_object)
        self.pushButton_brush_object.setObjectName(u"pushButton_brush_object")
        self.pushButton_brush_object.setGeometry(QRect(0, 10, 101, 31))
        self.pushButton_refresh_object = QPushButton(self.groupBox_label_object)
        self.pushButton_refresh_object.setObjectName(u"pushButton_refresh_object")
        self.pushButton_refresh_object.setGeometry(QRect(0, 440, 61, 31))
        self.groupBox_image_ori = QGroupBox(self.centralwidget)
        self.groupBox_image_ori.setObjectName(u"groupBox_image_ori")
        self.groupBox_image_ori.setGeometry(QRect(10, 232, 641, 471))
        self.groupBox_image_ori.setAutoFillBackground(False)
        self.groupBox_image_ori.setAlignment(Qt.AlignCenter)
        self.groupBox_image_ori.setFlat(False)
        self.label_image_ori = QLabel(self.groupBox_image_ori)
        self.label_image_ori.setObjectName(u"label_image_ori")
        self.label_image_ori.setGeometry(QRect(80, 20, 450, 450))
        self.label_image_ori.setMouseTracking(True)
        self.label_image_ori.setLineWidth(0)
        self.label_image_ori.setAlignment(Qt.AlignCenter)
        self.label_number_of_samples_4 = QLabel(self.groupBox_image_ori)
        self.label_number_of_samples_4.setObjectName(u"label_number_of_samples_4")
        self.label_number_of_samples_4.setGeometry(QRect(560, 130, 51, 16))
        self.textBrowser_number_of_objects_in_picture = QTextBrowser(self.groupBox_image_ori)
        self.textBrowser_number_of_objects_in_picture.setObjectName(u"textBrowser_number_of_objects_in_picture")
        self.textBrowser_number_of_objects_in_picture.setGeometry(QRect(560, 100, 51, 31))
        self.label_number_of_1_object_7 = QLabel(self.groupBox_image_ori)
        self.label_number_of_1_object_7.setObjectName(u"label_number_of_1_object_7")
        self.label_number_of_1_object_7.setGeometry(QRect(560, 190, 60, 16))
        self.textBrowser_index_sample = QTextBrowser(self.groupBox_image_ori)
        self.textBrowser_index_sample.setObjectName(u"textBrowser_index_sample")
        self.textBrowser_index_sample.setGeometry(QRect(560, 150, 51, 29))
        self.pushButton_delete_instance = QPushButton(self.centralwidget)
        self.pushButton_delete_instance.setObjectName(u"pushButton_delete_instance")
        self.pushButton_delete_instance.setEnabled(True)
        self.pushButton_delete_instance.setGeometry(QRect(1320, 610, 71, 31))
        self.textlabel_index_object = QTextEdit(self.centralwidget)
        self.textlabel_index_object.setObjectName(u"textlabel_index_object")
        self.textlabel_index_object.setGeometry(QRect(870, 650, 51, 31))
        self.label_20 = QLabel(self.centralwidget)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(840, 650, 32, 31))
        self.radioButton_create_dataset_from_images_dir = QRadioButton(self.centralwidget)
        self.radioButton_create_dataset_from_images_dir.setObjectName(u"radioButton_create_dataset_from_images_dir")
        self.radioButton_create_dataset_from_images_dir.setGeometry(QRect(10, 30, 81, 20))
        self.comboBox_dataset_list = QComboBox(self.centralwidget)
        self.comboBox_dataset_list.setObjectName(u"comboBox_dataset_list")
        self.comboBox_dataset_list.setGeometry(QRect(86, 120, 151, 31))
        self.pushButton_delete_dataset = QPushButton(self.centralwidget)
        self.pushButton_delete_dataset.setObjectName(u"pushButton_delete_dataset")
        self.pushButton_delete_dataset.setGeometry(QRect(390, 120, 81, 31))
        self.label_21 = QLabel(self.centralwidget)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(16, 130, 71, 16))
        self.label_21.setTextFormat(Qt.PlainText)
        self.label_21.setScaledContents(False)
        self.label_21.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(110, 90, 61, 16))
        self.textlabel_dataset_name = QTextEdit(self.centralwidget)
        self.textlabel_dataset_name.setObjectName(u"textlabel_dataset_name")
        self.textlabel_dataset_name.setGeometry(QRect(180, 80, 101, 31))
        self.pushButton_load_dataset_from_fiftyone_dataset = QPushButton(self.centralwidget)
        self.pushButton_load_dataset_from_fiftyone_dataset.setObjectName(u"pushButton_load_dataset_from_fiftyone_dataset")
        self.pushButton_load_dataset_from_fiftyone_dataset.setGeometry(QRect(290, 120, 91, 31))
        self.textlabel_path_image_dir = QTextEdit(self.centralwidget)
        self.textlabel_path_image_dir.setObjectName(u"textlabel_path_image_dir")
        self.textlabel_path_image_dir.setGeometry(QRect(180, 0, 331, 31))
        self.label_19 = QLabel(self.centralwidget)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(110, 10, 60, 16))
        self.label_19.setTextFormat(Qt.PlainText)
        self.label_19.setScaledContents(False)
        self.label_19.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.pushButton_create_dataset_from_image_dir = QPushButton(self.centralwidget)
        self.pushButton_create_dataset_from_image_dir.setObjectName(u"pushButton_create_dataset_from_image_dir")
        self.pushButton_create_dataset_from_image_dir.setGeometry(QRect(530, 80, 75, 31))
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(260, 750, 151, 16))
        self.label_14 = QLabel(self.centralwidget)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(10, 210, 75, 16))
        self.label_14.setTextFormat(Qt.PlainText)
        self.label_14.setWordWrap(False)
        self.label_14.setTextInteractionFlags(Qt.NoTextInteraction)
        self.textlabel_path_current_picture = QTextEdit(self.centralwidget)
        self.textlabel_path_current_picture.setObjectName(u"textlabel_path_current_picture")
        self.textlabel_path_current_picture.setGeometry(QRect(100, 200, 431, 31))
        self.radioButton_displayrbox = QRadioButton(self.centralwidget)
        self.radioButton_displayrbox.setObjectName(u"radioButton_displayrbox")
        self.radioButton_displayrbox.setGeometry(QRect(552, 296, 73, 20))
        self.radioButton_displayrbox.setAutoExclusive(False)
        self.radioButton_displaymask = QRadioButton(self.centralwidget)
        self.radioButton_displaymask.setObjectName(u"radioButton_displaymask")
        self.radioButton_displaymask.setGeometry(QRect(552, 244, 71, 20))
        self.radioButton_displaymask.setAutoExclusive(False)
        self.radioButton_displaybbox = QRadioButton(self.centralwidget)
        self.radioButton_displaybbox.setObjectName(u"radioButton_displaybbox")
        self.radioButton_displaybbox.setGeometry(QRect(552, 270, 77, 20))
        self.radioButton_displaybbox.setAutoExclusive(False)
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(933, 660, 32, 16))
        self.textlabel_label_name_1 = QTextEdit(self.centralwidget)
        self.textlabel_label_name_1.setObjectName(u"textlabel_label_name_1")
        self.textlabel_label_name_1.setGeometry(QRect(970, 650, 81, 31))
        self.textlabel_label_name_2 = QTextEdit(self.centralwidget)
        self.textlabel_label_name_2.setObjectName(u"textlabel_label_name_2")
        self.textlabel_label_name_2.setGeometry(QRect(1097, 650, 81, 31))
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(1203, 660, 32, 16))
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(1060, 660, 32, 16))
        self.textlabel_label_name_3 = QTextEdit(self.centralwidget)
        self.textlabel_label_name_3.setObjectName(u"textlabel_label_name_3")
        self.textlabel_label_name_3.setGeometry(QRect(1240, 650, 91, 31))
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(940, 700, 158, 26))
        self.horizontalLayout_2 = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.pushButton_front_object = QPushButton(self.layoutWidget)
        self.pushButton_front_object.setObjectName(u"pushButton_front_object")

        self.horizontalLayout_2.addWidget(self.pushButton_front_object)

        self.pushButton_next_object = QPushButton(self.layoutWidget)
        self.pushButton_next_object.setObjectName(u"pushButton_next_object")

        self.horizontalLayout_2.addWidget(self.pushButton_next_object)

        self.layoutWidget1 = QWidget(self.centralwidget)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(28, 710, 421, 26))
        self.horizontalLayout = QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton_label_save_new_object = QPushButton(self.layoutWidget1)
        self.pushButton_label_save_new_object.setObjectName(u"pushButton_label_save_new_object")

        self.horizontalLayout.addWidget(self.pushButton_label_save_new_object)

        self.pushButton_front_picture = QPushButton(self.layoutWidget1)
        self.pushButton_front_picture.setObjectName(u"pushButton_front_picture")

        self.horizontalLayout.addWidget(self.pushButton_front_picture)

        self.pushButton_next_picture = QPushButton(self.layoutWidget1)
        self.pushButton_next_picture.setObjectName(u"pushButton_next_picture")

        self.horizontalLayout.addWidget(self.pushButton_next_picture)

        self.pushButton_delete_image = QPushButton(self.layoutWidget1)
        self.pushButton_delete_image.setObjectName(u"pushButton_delete_image")
        self.pushButton_delete_image.setCheckable(False)

        self.horizontalLayout.addWidget(self.pushButton_delete_image)

        self.pushButton_expand_the_dataset = QPushButton(self.centralwidget)
        self.pushButton_expand_the_dataset.setObjectName(u"pushButton_expand_the_dataset")
        self.pushButton_expand_the_dataset.setGeometry(QRect(480, 120, 75, 31))
        self.label_22 = QLabel(self.centralwidget)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(110, 50, 71, 20))
        self.label_22.setTextFormat(Qt.PlainText)
        self.label_22.setScaledContents(False)
        self.label_22.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.textlabel_path_label_file = QTextEdit(self.centralwidget)
        self.textlabel_path_label_file.setObjectName(u"textlabel_path_label_file")
        self.textlabel_path_label_file.setGeometry(QRect(180, 40, 331, 31))
        self.pushButton_label_file_to_load = QPushButton(self.centralwidget)
        self.pushButton_label_file_to_load.setObjectName(u"pushButton_label_file_to_load")
        self.pushButton_label_file_to_load.setGeometry(QRect(530, 40, 75, 31))
        self.pushButton_image_dir_to_load = QPushButton(self.centralwidget)
        self.pushButton_image_dir_to_load.setObjectName(u"pushButton_image_dir_to_load")
        self.pushButton_image_dir_to_load.setGeometry(QRect(530, 0, 75, 31))
        self.pushButton_load_special_image = QPushButton(self.centralwidget)
        self.pushButton_load_special_image.setObjectName(u"pushButton_load_special_image")
        self.pushButton_load_special_image.setGeometry(QRect(550, 200, 91, 31))
        self.pushButton_display_on_web = QPushButton(self.centralwidget)
        self.pushButton_display_on_web.setObjectName(u"pushButton_display_on_web")
        self.pushButton_display_on_web.setGeometry(QRect(560, 120, 91, 31))
        self.comboBox_class_filer = QComboBox(self.centralwidget)
        self.comboBox_class_filer.setObjectName(u"comboBox_class_filer")
        self.comboBox_class_filer.setGeometry(QRect(90, 160, 151, 31))
        self.label_23 = QLabel(self.centralwidget)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(20, 170, 51, 16))
        self.label_23.setTextFormat(Qt.PlainText)
        self.label_23.setScaledContents(False)
        self.label_23.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.pushButton_select_running = QPushButton(self.centralwidget)
        self.pushButton_select_running.setObjectName(u"pushButton_select_running")
        self.pushButton_select_running.setGeometry(QRect(250, 160, 91, 31))
        self.pushButton_update = QPushButton(self.centralwidget)
        self.pushButton_update.setObjectName(u"pushButton_update")
        self.pushButton_update.setGeometry(QRect(240, 120, 41, 31))
        self.radioButton_locklabel = QRadioButton(self.centralwidget)
        self.radioButton_locklabel.setObjectName(u"radioButton_locklabel")
        self.radioButton_locklabel.setGeometry(QRect(1340, 660, 71, 20))
        self.radioButton_locklabel.setAutoExclusive(False)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1635, 22))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u753b\u7b14\u5c3a\u5bf8", None))
        self.pushButton_save_object.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u5e76\u4e0b\u4e00\u4e2a", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"\u6807\u6ce8\u8fdb\u5ea6\uff1a", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"\u4f5c\u8005\uff1aLance_Shao  \u54c8\u5c14\u6ee8\u5de5\u4e1a\u5927\u5b66  QQ:261983626 CSDN: https://blog.csdn.net/scy261983626", None))
        self.label_state_feedback.setText(QCoreApplication.translate("MainWindow", u"\u8f6f\u4ef6\u72b6\u6001\u63d0\u793a:", None))
        self.label_label_number_process.setText(QCoreApplication.translate("MainWindow", u"\u5f53\u524d\u6570/\u603b\u6570", None))
        self.pushButton_save_picture.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u5e76\u4e0b\u4e00\u5f20", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u5feb\u6377\u952e\uff1a  1:\u7269\u54c1\u753b\u7b14 2\uff1a\u80cc\u666f\u753b\u7b14 3:\u91cd\u65b0\u7ed8\u5236  4\uff1a\u4fdd\u5b58  \u5de6\u952e\u4e3a\u7269\u54c1\u753b\u7b14\uff0c\u53f3\u952e\u4e3a\u80cc\u666f\u753b\u7b14  ", None))
        self.label_number_of_1_object_2.setText(QCoreApplication.translate("MainWindow", u"2-5\u4f8b\u6570\uff1a", None))
        self.textBrowser_num_2_5_object.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>", None))
        self.label_number_of_1_object_3.setText(QCoreApplication.translate("MainWindow", u"6-9\u4f8b\u6570\uff1a", None))
        self.textBrowser_num_6_9_object.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>", None))
        self.label_number_of_1_object_4.setText(QCoreApplication.translate("MainWindow", u"9-13\u4f8b\u6570\uff1a", None))
        self.textBrowser_num_10_13_object.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>", None))
        self.label_number_of_samples.setText(QCoreApplication.translate("MainWindow", u"\u6837\u672c\u603b\u6570\uff1a", None))
        self.textBrowser_sum_images.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">1</span></p></body></html>", None))
        self.label_number_of_label_samples.setText(QCoreApplication.translate("MainWindow", u"\u5df2\u6807\u8bb0\u6837\u672c\u6570\uff1a", None))
        self.textBrowser_num_labeled.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>", None))
        self.label_number_of_1_object.setText(QCoreApplication.translate("MainWindow", u"1\u4f8b\u6570\uff1a", None))
        self.textBrowser_num_1_object.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>", None))
        self.groupBox_label_object.setTitle(QCoreApplication.translate("MainWindow", u"\u6807\u6ce8\u533a\u57df", None))
        self.label_image_object.setText("")
        self.pushButton_brush_background.setText(QCoreApplication.translate("MainWindow", u"\u80cc\u666f\u753b\u7b14", None))
        self.pushButton_brush_object.setText(QCoreApplication.translate("MainWindow", u"\u7269\u54c1\u753b\u7b14", None))
        self.pushButton_refresh_object.setText(QCoreApplication.translate("MainWindow", u"\u91cd\u7ed8", None))
        self.groupBox_image_ori.setTitle(QCoreApplication.translate("MainWindow", u"\u5f53\u524d\u56fe\u7247\u6837\u672c\u663e\u793a\u533a", None))
        self.label_image_ori.setText("")
        self.label_number_of_samples_4.setText(QCoreApplication.translate("MainWindow", u"\u5b9e \u4f8b \u6570 ", None))
        self.textBrowser_number_of_objects_in_picture.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>", None))
        self.label_number_of_1_object_7.setText(QCoreApplication.translate("MainWindow", u"\u7b2c\u51e0\u5f20\u56fe\u7247", None))
        self.textBrowser_index_sample.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.pushButton_delete_instance.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664\u5b9e\u4f8b", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"\u5e8f\u53f7", None))
        self.radioButton_create_dataset_from_images_dir.setText(QCoreApplication.translate("MainWindow", u"\u521b\u5efa\u6570\u636e\u96c6", None))
        self.pushButton_delete_dataset.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664\u6570\u636e\u5e93", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"\u73b0\u6709\u6570\u636e\u96c6", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u5e93\u540d\u79f0", None))
        self.pushButton_load_dataset_from_fiftyone_dataset.setText(QCoreApplication.translate("MainWindow", u"\u8f7d\u5165\u6570\u636e\u5e93", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u7247\u6587\u4ef6\u5939", None))
        self.pushButton_create_dataset_from_image_dir.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u5e76\u521b\u5efa", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u5feb\u6377\u952e\uff1a4\uff1a\u4fdd\u5b58\u65b0\u5efa\u5b9e\u4f8b", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"\u5f53\u524d\u56fe\u7247\u8def\u5f84:", None))
        self.radioButton_displayrbox.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793arbox", None))
        self.radioButton_displaymask.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u5206\u5272", None))
        self.radioButton_displaybbox.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793abbox", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e1", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e3", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e2", None))
        self.pushButton_front_object.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4e00\u4e2a", None))
        self.pushButton_next_object.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u4e00\u4e2a", None))
        self.pushButton_label_save_new_object.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u65b0\u5b9e\u4f8b", None))
        self.pushButton_front_picture.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4e00\u5f20", None))
        self.pushButton_next_picture.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u4e00\u5f20", None))
        self.pushButton_delete_image.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664", None))
        self.pushButton_expand_the_dataset.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u6570\u636e\u96c6", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e\u6587\u4ef6\u5939", None))
        self.pushButton_label_file_to_load.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u76ee\u5f55", None))
        self.pushButton_image_dir_to_load.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u76ee\u5f55", None))
        self.pushButton_load_special_image.setText(QCoreApplication.translate("MainWindow", u"\u8f7d\u5165\u6307\u5b9a\u56fe\u7247", None))
        self.pushButton_display_on_web.setText(QCoreApplication.translate("MainWindow", u"\u7f51\u9875\u7aef\u9884\u89c8", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"\u7c7b\u522b\u7b5b\u9009", None))
        self.pushButton_select_running.setText(QCoreApplication.translate("MainWindow", u"\u6267\u884c\u7b5b\u9009", None))
        self.pushButton_update.setText(QCoreApplication.translate("MainWindow", u"\u5237\u65b0", None))
        self.radioButton_locklabel.setText(QCoreApplication.translate("MainWindow", u"\u9501\u5b9a\u6807\u7b7e", None))
    # retranslateUi

