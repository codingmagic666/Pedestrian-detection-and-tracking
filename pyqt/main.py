# Copyright © 2020, Yingping Liang. All Rights Reserved.

# Copyright Notice
# Yingping Liang copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yingping Liang.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yingping Liang assumes no responsibility for any errors contained herein.

import sys
import argparse
import qdarkstyle
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication
from UILib.MainWindow import MainWindow
import os
import PySide2
import PyQt5_stylesheets


def main(opt):
    '''
    启动PyQt5程序，打开GUI界面
    '''
    app = QApplication(sys.argv)
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap("data/logo.png"))
    splash.showMessage("加载... 0%", QtCore.Qt.AlignHCenter, QtCore.Qt.black)
    splash.show()                           # 显示启动界面
    QtWidgets.qApp.processEvents()          # 处理主进程事件
    main_window = MainWindow(opt)
    main_window.load_data(splash)           # 加载数据
    # main_window.showFullScreen()          # 全拼显示
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    #app.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style='style_Classic'))
    splash.close()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str,
                        default='torch', help='model framework.')
    parser.add_argument('--tracker', type=str,
                        default='deep_sort', help='tracker framework.')
    opt = parser.parse_args()

    main(opt)
