# Copyright © 2020, Yingping Liang. All Rights Reserved.

# Copyright Notice
# Yingping Liang copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yingping Liang.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yingping Liang assumes no responsibility for any errors contained herein.

import os
import cv2
import time
from PyQt5 import QtCore, QtWidgets, QtMultimedia
from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtCore import QRect, Qt
import imutils
from PyQt5.QtGui import QImage, QPixmap, QFont, QPen, QPainter
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QListWidget, QAction
from PyQt5.QtWidgets import qApp, QMenu, QVBoxLayout, QFileDialog, QLabel
from PyQt5.uic import loadUi
from UILib.ViolationItem import ViolationItem
from processor.MainProcessor import MainProcessor

class MainWindowLayOut(QMainWindow):
    def __init__(self, opt):

        super(MainWindowLayOut, self).__init__()
        loadUi("./data/UI/MainWindow.ui", self)

        self.face_count = 0
        self.max_log_num = 5
        self.face_size = 140
        self.start_img_x = self.live_preview.geometry().x()
        self.start_img_y = self.live_preview.geometry().y()
        self.feed = None
        self.vs = None
        self.opt = opt
        self.wave_file = './data/warning.wav'
        self.updateCamInfo()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        #self.statusBar.showMessage("欢迎")

        self.clear_button.clicked.connect(self.clear)
        self.refresh_button.clicked.connect(self.refresh)

        font = QFont()
        font.setPointSize(10)
        
        self.log_tabwidget.clear()
        self.log_tabwidget.setFont(font)
        self.ip_btn.setFont(font)
        self.file_btn.setFont(font)
        self.face_list = QListWidget(self)
        self.illegal_list = QListWidget(self)
        self.longstay_list = QListWidget(self)
        self.log_tabwidget.addTab(self.face_list, "行人记录")
        self.log_tabwidget.addTab(self.longstay_list, "行人逗留记录")
        self.log_tabwidget.addTab(self.illegal_list, "区域入侵记录")

        self.initParams()
        #self.initMenu(font)

    def initParams(self):

        menubar = self.menuBar()
        #settingsMenu = menubar.addMenu('&设置')
        #self.add_setting_menu(settingsMenu)
        self.ip_edit.setText('192.168.1.212')
        #self.file_edit.setText('/home/tarzan/cc/qt/project/1.mp4')
        self.file_btn.clicked.connect(
            lambda: self.getFile(self.file_edit))  # 文件选择槽函数绑定
        self.ip_btn.clicked.connect(
            lambda: self.updateCamInfo(self.ip_edit.text()))  # 文件选择槽函数绑定
        # self.region_bbox2draw = [['region'], ['wall']]
        self.region_bbox2draw = [[], []]
        self.values_max_num = 30
        file = QUrl.fromLocalFile(self.wave_file)  # 音频文件路径
        content = QtMultimedia.QMediaContent(file)
        self.wav_player = QtMultimedia.QMediaPlayer()
        self.wav_player.setMedia(content)
        self.wav_player.setVolume(30.0)

    def add_setting_menu(self, settingsMenu):

        speed_menu = QMenu("更改模型", self)
        settingsMenu.addMenu(speed_menu)

        act = QAction('YOLOv3+DarkNet53', self)
        act.setStatusTip('YOLOv3+DarkNet53')
        speed_menu.addAction(act)

        act = QAction('YOLOv3+MobileNetV3', self)
        act.setStatusTip('YOLOv3+MobileNetV3')
        speed_menu.addAction(act)

        direct_menu = QMenu("更改阈值", self)
        settingsMenu.addMenu(direct_menu)

        act = QAction('阈值+0.05', self)
        act.setStatusTip('阈值+0.05')
        direct_menu.addAction(act)

        act = QAction('阈值-0.05', self)
        act.setStatusTip('阈值-0.05')
        direct_menu.addAction(act)

    def initMenu(self, font):

        menubar = self.menuBar()
        menubar.setFont(font)
        fileMenu = menubar.addMenu('&文件')

        addRec = QMenu("添加记录", self)

        act = QAction('添加摄像头', self)
        act.setStatusTip('Add Camera Manually')
        addRec.addAction(act)

        act = QAction('添加记录', self)
        act.setStatusTip('Add Car Manually')
        addRec.addAction(act)

        fileMenu.addMenu(addRec)

        act = QAction('&存档', self)
        act.setStatusTip('Show Archived Records')
        fileMenu.addAction(act)

        # Add Exit
        fileMenu.addSeparator()
        act = QAction('&退出', self)
        act.setStatusTip('退出应用')
        act.triggered.connect(qApp.quit)
        fileMenu.addAction(act)

    def updateCamInfo(self,feed='1.mp4'):

        self.region_bbox2draw = [[], []]
        self.feed = feed
        self.processor = MainProcessor(
            model_type=self.opt.model_type, tracker_type=self.opt.tracker)
        self.vs = cv2.VideoCapture(self.feed)
        #self.vs.set(cv2.CAP_PROP_FRAME_WIDTH,600)
        #self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT,700)


    def getFile(self, lineEdit):
        file_path = QFileDialog.getOpenFileName()[0]
        lineEdit.setText(file_path)  # 获取文件路径
        self.updateCamInfo(file_path)

    def updateLog(self, data=[]):

        for row in data:
            if self.face_count < self.max_log_num:
                self.face_count += 1
            else:
                self.face_count = 0
                self.face_list.clear()
            listWidget = ViolationItem()
            listWidget.setData(row)
            listWidgetItem = QtWidgets.QListWidgetItem(self.face_list)
            listWidgetItem.setSizeHint(listWidget.sizeHint())
            self.face_list.addItem(listWidgetItem)
            self.face_list.setItemWidget(listWidgetItem, listWidget)
            if not row['RULENAME'] == '正常行为':
                if row['RULENAME'] == 'douliu':
                    listWidget = ViolationItem()
                    listWidget.setData(row)
                    listWidgetItem = QtWidgets.QListWidgetItem(self.longstay_list)
                    listWidgetItem.setSizeHint(listWidget.sizeHint())
                    self.longstay_list.addItem(listWidgetItem)
                    self.longstay_list.setItemWidget(listWidgetItem, listWidget)
                if row['RULENAME'] == '危险行为-':
                    listWidget = ViolationItem()
                    listWidget.setData(row)
                    listWidgetItem = QtWidgets.QListWidgetItem(self.illegal_list)
                    listWidgetItem.setSizeHint(listWidget.sizeHint())
                    self.illegal_list.addItem(listWidgetItem)
                    self.illegal_list.setItemWidget(listWidgetItem, listWidget)

    @QtCore.pyqtSlot()
    def refresh(self):
        self.updateCamInfo("rtsp://admin:neu307neu307@192.168.1.201")

    @QtCore.pyqtSlot()
    def clear(self):
        qm = QtWidgets.QMessageBox
        prompt = qm.question(self, '', "是否重置所有记录?", qm.Yes | qm.No)
        if prompt == qm.Yes:
            self.face_list.clear()
            self.illegal_list.clear()
            self.longstay_list.clear()
        else:
            pass

    def toQImage(self, img, height=800):
        if not height is None:
            img = imutils.resize(img, height=height)
        for i, bbox in enumerate(self.region_bbox2draw):
            if len(bbox) > 2:
                x1, y1 = bbox[0]
                x0, y0 = x1, y1
                color = (255, 0, 0)
                for v in bbox[1:]:
                    x2, y2 = v
                    if i == 0:
                        color = (255, 0, 0)
                    else:
                        color = (255, 255, 0)
                    cv2.line(img, (x1, y1), (x2, y2), color, 2)
                    x1, y1 = x2, y2
                cv2.line(img, (x0, y0), (x1, y1), color, 2)

        h, w = img.shape[:2]
        img_x = self.live_preview.geometry().x()
        self.start_img_x = img_x + \
            int(self.live_preview.geometry().width()/2 - w/2)
        img_y = self.live_preview.geometry().y()
        self.start_img_y = img_y + \
            int(self.live_preview.geometry().height()/2 - h/2)

        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(
            img.tobytes(), img.shape[1], img.shape[0], img.strides[0], qformat)
        outImg = outImg.rgbSwapped()

        return outImg

    def load_data(self, sp):
        for i in range(1, 11):  # 模拟主程序加载过程
            time.sleep(0.5)                   # 加载数据
            sp.showMessage("加载... {0}%".format(
                i * 10), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.black)
            QtWidgets.qApp.processEvents()  # 允许主进程处理事件

    def mouseReleaseEvent(self, event):  # 鼠标键释放时调用
        # 参数1：鼠标的作用对象；参数2：鼠标事件对象，用来保存鼠标数据
        self.unsetCursor()
        n = event.button()  # 用来判断是哪个鼠标健触发了事件【返回值：0  1  2  4】
        if n == 1:
            if self.region_checkbtn.isChecked(): #or self.wall_checkbtn.isChecked():
                x = event.x()  # 返回鼠标相对于窗口的x轴坐标
                y = event.y()  # 返回鼠标相对于窗口的y轴坐标
                print('-[INFO] Mouse Released', n, x -
                      self.start_img_x, y - self.start_img_y)
                if self.region_checkbtn.isChecked():
                    self.region_bbox2draw[0].append(
                        [x - self.start_img_x, y - self.start_img_y - 20])
                else:
                    self.region_bbox2draw[1].append(
                        [x - self.start_img_x, y - self.start_img_y - 20])
