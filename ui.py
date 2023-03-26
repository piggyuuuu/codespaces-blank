import argparse
import platform
import shutil
import time
from numpy import random
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np
import time

def load_model(
        weights=ROOT / 'best.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference

):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    return model, stride, names, pt, jit, onnx, engine


def run(model, img, stride, pt,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.35,  # confidence threshold
        iou_thres=0.15,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        ):

    cal_detect = []

    device = select_device(device)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Set Dataloader
    im = letterbox(img, imgsz, stride, pt)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=augment)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

            # Write results

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]}'
                lbl = names[int(cls)]

                cal_detect.append([label, xyxy,float(conf),c])
    return cal_detect


def det_yolov7(info1):
    global model, stride, names, pt, jit, onnx, engine
    list = ['人', '自行车', '车', '摩托车', '飞机', '公共汽车', '火车', '卡车', '船', '红绿灯', '消火栓', '停车标志', '停车收费表', '板凳', '鸟', '猫', '狗','马', '羊', '牛', '大象', \
            '熊', '斑马', '长颈鹿', '背包', '雨伞', '手袋', '领带', '手提箱', '飞盘', '滑雪板', '滑雪板', '运动球', '风筝', '棒球棒', '棒球手套', '滑板','冲浪板', '网球拍', \
            '瓶', '酒杯', '杯', '叉子', '刀', '汤匙', '碗', '香蕉', '苹果', '三明治', '橙子', '西兰花', '胡萝卜', '热狗', '披萨', '甜甜圈', '蛋糕', '椅子','沙发', \
            '盆栽的植物', '床', '餐桌', '厕所', '电视', '笔记本电脑', '鼠标', '遥控器', '键盘', '手机', '微波炉', '烤箱', '烤面包机', '水槽', '冰箱', '书','时钟', '花瓶', '剪刀', '泰迪熊', '吹风机', '牙刷']
    if info1[-3:] in ['jpg','png','jpeg','tif','bmp']:
        image = cv2.imread(info1)  # 读取识别对象
        #try:
        results = run(model, image, stride, pt)  # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
        for i in results:
            box = i[1]
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            color = [255, 0, 0]
            ui.printf(list[i[3]])
            with open('./save.txt','a',encoding='utf-8') as f:
                f.write(list[i[3]] + '\n')
            cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        #except:
            #pass
        ui.showimg(image)
    if info1[-3:] in ['mp4','avi']:
        capture = cv2.VideoCapture(info1)
        while True:
            _, image = capture.read()
            if image is None:
                break
            try:
                results = run(model, image, stride, pt)  # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
                for i in results:
                    box = i[1]
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    color = [255, 0, 0]
                    ui.printf(list[i[3]])
                    with open('./save.txt', 'a', encoding='utf-8') as f:
                        f.write(list[i[3]] + '\n')
                    cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                    cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            except:
                pass
            ui.showimg(image)
            QApplication.processEvents()

class Thread_1(QThread):  # 线程1
    def __init__(self,info1):
        super().__init__()
        self.info1=info1
        self.run2(self.info1)

    def run2(self, info1):
        result = []
        result = det_yolov7(info1)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/carui.png\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(168, 60, 491, 71))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-size:50px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0.6);")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 188, 751, 501))
        self.label_2.setStyleSheet("background:rgba(255,255,255,1);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(73, 746, 851, 174))
        self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.8);")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1020, 750, 150, 40))
        self.pushButton.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1020, 810, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1020, 870, 150, 40))
        self.pushButton_3.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像识别系统"))
        self.label.setText(_translate("MainWindow", "图像识别系统"))
        self.label_2.setText(_translate("MainWindow", "请点击以添加文件"))
        self.pushButton.setText(_translate("MainWindow", "添加文件"))
        self.pushButton_2.setText(_translate("MainWindow", "开始检测"))
        self.pushButton_3.setText(_translate("MainWindow", "退出系统"))

        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.handleCalc3)

    def openfile(self):
        global sname, filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("当前选择的文件路径是：%s" % filepath)


    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 800
        else:
            ratio = n_height / 800
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_2.setPixmap(QPixmap.fromImage(new_img))

    def click_1(self):
        global filepath
        try:
            self.thread_1.quit()
        except:
            pass
        self.thread_1 = Thread_1(filepath)  # 创建线程
        self.thread_1.wait()
        self.thread_1.start()  # 开始线程


if __name__ == "__main__":
    global model, stride, names, pt, jit, onnx, engine
    model, stride, names, pt, jit, onnx, engine = load_model()  # 加载模型
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
