### Pedestrian-detection-and-tracking

## 这是一个行人检测与跟踪的项目

### YOLOv3-cbam-rfb文件夹介绍

这里提供了几个核心文件，主要作用为

+ cfg文件夹                       ：提供YOLOv3网络结构的cfg文件，其中yolov3-cbam-rfb.cfg为本文最终训练时使用的cfg文件
+ Anchor.py                      ：提供了使用Kmeans得到在某个数据集下先验框的脚本
+ labels.py                         ：提供了将数据集中xml文件转换为txt文件的脚本
+ voc_annotation.py        ：提供了数据集划分脚本

### pyqt文件夹介绍

这里提供了行人检测与跟踪系统的核心文件

+ UILib文件夹                       ：处理Qt Designer生成的ui文件
+ data文件夹                        ：系统的客户端界面文件，包括ui以及图片，测试程序等
+ processor文件夹              ：将YOLOv3以及Deep-Sort算法封装为类，便于系统调用
+ main.py                             ：软件系统启动程序
+ main.sh                            ：软件系统启动脚本，包括激活虚拟环境，指定GPU序号以及启动程序

#### 测试视频

 [![Watch the video](https://github.com/codingmagic666/Pedestrian-detection-and-tracking/blob/main/extra/display.gif)]()

