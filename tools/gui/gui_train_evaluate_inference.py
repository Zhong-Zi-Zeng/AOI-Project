from PyQt6 import QtWidgets, QtCore
import sys
import os
import subprocess
import yaml


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI-Training & Evaluation & Inference')
        self.resize(900, 1000)
        self.setStyleSheet('background:#1F2B37')
        self.ui()

    def ui(self):
        # ====================config file====================
        # model
        self.label1 = QtWidgets.QLabel(self)  # Label
        self.label1.setText('Model used:')
        self.label1.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label1.move(5, 20)
        self.box1 = QtWidgets.QComboBox(self)  # 下拉選單
        self.box1.addItems(['YOLOv7-Object Detection',
                            'YOLOv7-Instance Segmentation',
                            'Cascade-Mask RCNN',
                            'Mask2Former'])
        self.box1.setCurrentIndex(0)  # default, model=YOLOv7-Object Detection
        self.box1.setGeometry(300, 20, 300, 30)
        self.box1.setStyleSheet('''
                                QComboBox {
                                    color: #DEDEDE;
                                    font-size: 18px;
                                    font-weight: bold;
                                    font-family: monospace;
                                }
                                QComboBox:disabled {
                                    color: #808080;
                                }
                            ''')
        self.box1.view().setStyleSheet('''  
                    QAbstractItemView {
                        color: #DEDEDE;
                        background-color: #1F2B37;
                    }
                ''')  # 解決QComboBox時，無法實現 color: #DEDEDE
        self.box1.currentTextChanged.connect(self.enable_disable_controls)  # 控制 backbone

        # backbone
        self.label2 = QtWidgets.QLabel(self)  # Label
        self.label2.setText('Backbone used:')
        self.label2.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label2.move(5, 60)
        self.box2 = QtWidgets.QComboBox(self)  # 下拉選單
        self.box2.addItems(['ResNet-50',
                            'ResNet-101',
                            'ResNeXt-101',
                            'Swin Transformer'])
        self.box2.setCurrentIndex(0)  # default, model=YOLOv7-Object Detection 不可選 backbone
        self.box2.setDisabled(True)
        self.box2.setGeometry(300, 60, 300, 30)
        self.box2.setStyleSheet('''
                                QComboBox {
                                    color: #DEDEDE;
                                    font-size: 18px;
                                    font-weight: bold;
                                    font-family: monospace;
                                }
                                QComboBox:disabled {
                                    color: #808080;
                                }
                            ''')
        self.box2.view().setStyleSheet('''  
                            QAbstractItemView {
                                color: #DEDEDE;
                                background-color: #1F2B37;
                            }
                        ''')  # 解決QComboBox時，無法實現 color: #DEDEDE

        # coco_root
        self.label3 = QtWidgets.QLabel(self)  # Label
        self.label3.setText('Path to coco dataset:')
        self.label3.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label3.move(5, 100)
        self.input3 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input3.setGeometry(300, 100, 300, 30)
        self.input3.setStyleSheet('''
                                            background:#DEDEDE;
                                            color:#1F2B37;
                                            font-size:18px;
                                            font-weight:bold;
                                            font-family:monospace;
                                            ''')
        self.btn3 = QtWidgets.QPushButton(self)  # Button
        self.btn3.move(650, 100)
        self.btn3.setText('Browse')
        self.btn3.setStyleSheet('''
                                        background:#77F2A1;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.btn3.clicked.connect(lambda: self.openFolder(self.input3))

        # training dataset：folder -> yolov7-inSeg
        self.label4 = QtWidgets.QLabel(self)  # Label
        self.label4.setText('Path to training folder:')
        self.label4.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label4.move(5, 140)
        self.input4 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input4.setGeometry(300, 140, 300, 30)
        self.input4.setStyleSheet('''
                                QLineEdit {
                                    background:#DEDEDE;
                                    color:#1F2B37;
                                    font-size:18px;
                                    font-weight:bold;
                                    font-family:monospace;
                                }
                                QLineEdit:disabled {
                                    background:#808080;
                                }
                                    ''')
        self.input4.setDisabled(True)   # default：yolov7-obj
        self.btn4 = QtWidgets.QPushButton(self)  # Button
        self.btn4.move(650, 140)
        self.btn4.setText('Browse')
        self.btn4.setStyleSheet('''
                            QPushButton{
                                background:#77F2A1;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                            }
                            QPushButton:disabled {
                                background:#808080;
                                color:#DEDEDE;
                            }
                            ''')
        self.btn4.clicked.connect(lambda: self.openFolder(self.input4))
        self.btn4.setDisabled(True)     # default：yolov7-obj

        # test dataset：folder -> yolov7-inSeg
        self.label5 = QtWidgets.QLabel(self)  # Label
        self.label5.setText('Path to test folder:')
        self.label5.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label5.move(5, 180)
        self.input5 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input5.setGeometry(300, 180, 300, 30)
        self.input5.setStyleSheet('''
                                QLineEdit {
                                    background:#DEDEDE;
                                    color:#1F2B37;
                                    font-size:18px;
                                    font-weight:bold;
                                    font-family:monospace;
                                }
                                QLineEdit:disabled {
                                    background:#808080;
                                }
                                    ''')
        self.input5.setDisabled(True)   # default：yolov7-obj
        self.btn5 = QtWidgets.QPushButton(self)  # Button
        self.btn5.move(650, 180)
        self.btn5.setText('Browse')
        self.btn5.setStyleSheet('''
                            QPushButton{
                                background:#77F2A1;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                            }
                            QPushButton:disabled {
                                background:#808080;
                                color:#DEDEDE;
                            }
                                ''')
        self.btn5.clicked.connect(lambda: self.openFolder(self.input5))
        self.btn5.setDisabled(True)     # default：yolov7-obj

        # training dataset：txt -> yolov7-obj
        self.label4_2 = QtWidgets.QLabel(self)  # Label
        self.label4_2.setText('Path to train txt file:')
        self.label4_2.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label4_2.move(5, 220)
        self.input4_2 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input4_2.setGeometry(300, 220, 300, 30)
        self.input4_2.setStyleSheet('''
                                        QLineEdit {
                                            background:#DEDEDE;
                                            color:#1F2B37;
                                            font-size:18px;
                                            font-weight:bold;
                                            font-family:monospace;
                                        }
                                        QLineEdit:disabled {
                                            background:#808080;
                                        }
                                            ''')
        self.btn4_2 = QtWidgets.QPushButton(self)  # Button
        self.btn4_2.move(650, 220)
        self.btn4_2.setText('Browse')
        self.btn4_2.setStyleSheet('''
                                    QPushButton{
                                        background:#77F2A1;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                    }
                                    QPushButton:disabled {
                                        background:#808080;
                                        color:#DEDEDE;
                                    }
                                    ''')
        self.btn4_2.clicked.connect(lambda: self.openFile(self.input4_2))

        # test dataset：txt -> yolov7-obj
        self.label5_2 = QtWidgets.QLabel(self)  # Label
        self.label5_2.setText('Path to test txt file:')
        self.label5_2.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label5_2.move(5, 260)
        self.input5_2 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input5_2.setGeometry(300, 260, 300, 30)
        self.input5_2.setStyleSheet('''
                                        QLineEdit {
                                            background:#DEDEDE;
                                            color:#1F2B37;
                                            font-size:18px;
                                            font-weight:bold;
                                            font-family:monospace;
                                        }
                                        QLineEdit:disabled {
                                            background:#808080;
                                        }
                                            ''')
        self.btn5_2 = QtWidgets.QPushButton(self)  # Button
        self.btn5_2.move(650, 260)
        self.btn5_2.setText('Browse')
        self.btn5_2.setStyleSheet('''
                                    QPushButton{
                                        background:#77F2A1;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                    }
                                    QPushButton:disabled {
                                        background:#808080;
                                        color:#DEDEDE;
                                    }
                                        ''')
        self.btn5_2.clicked.connect(lambda: self.openFile(self.input5_2))

        # classes_yaml
        self.label7 = QtWidgets.QLabel(self)  # Label
        self.label7.setText('Path to classes.yaml file:')
        self.label7.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label7.move(5, 300)
        self.input7 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input7.setGeometry(300, 300, 300, 30)
        self.input7.setStyleSheet('''
                                        background:#DEDEDE;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.btn7 = QtWidgets.QPushButton(self)  # Button
        self.btn7.move(650, 300)
        self.btn7.setText('Browse')
        self.btn7.setStyleSheet('''
                                        background:#77F2A1;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.btn7.clicked.connect(lambda: self.openFile(self.input7))

        # Hyperparameter
        self.label6 = QtWidgets.QLabel(self)  # Label
        self.label6.setText('Hyperparameter settings:')
        self.label6.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6.move(5, 340)

        # optimizer
        self.label6_1 = QtWidgets.QLabel(self)  # Label
        self.label6_1.setText('optimizer:')
        self.label6_1.setStyleSheet('''
                                    color:#DEDEDE;
                                    font-size:18px;
                                    font-weight:bold;
                                    font-family:monospace;
                                    ''')
        self.label6_1.move(50, 380)
        self.box6_1 = QtWidgets.QComboBox(self)  # 下拉選單
        self.box6_1.addItems(['Adam', 'AdamW', 'SGD'])
        self.box6_1.setCurrentIndex(0)  # default, Stride=1
        self.box6_1.setGeometry(500, 380, 100, 30)
        self.box6_1.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')

        # weight
        self.label6_2 = QtWidgets.QLabel(self)  # Label
        self.label6_2.setText('weight file path:')
        self.label6_2.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_2.move(50, 420)
        self.input6_2 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_2.setGeometry(300, 420, 300, 30)
        self.input6_2.setStyleSheet('''
                                            background:#DEDEDE;
                                            color:#1F2B37;
                                            font-size:18px;
                                            font-weight:bold;
                                            font-family:monospace;
                                            ''')
        self.btn6_2 = QtWidgets.QPushButton(self)  # Button
        self.btn6_2.move(650, 420)
        self.btn6_2.setText('Browse')
        self.btn6_2.setStyleSheet('''
                                        background:#77F2A1;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.btn6_2.clicked.connect(lambda: self.openFile(self.input6_2))

        # start_epoch
        self.label6_3 = QtWidgets.QLabel(self)  # Label
        self.label6_3.setText('start epoch:')
        self.label6_3.setStyleSheet('''
                                    color:#DEDEDE;
                                    font-size:18px;
                                    font-weight:bold;
                                    font-family:monospace;
                                    ''')
        self.label6_3.move(50, 460)
        self.input6_3 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_3.setText('0')  # default, start_epoch=0
        self.input6_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_3.setGeometry(500, 460, 100, 30)
        self.input6_3.setStyleSheet('''
                                        background:#DEDEDE;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')

        # end_epoch
        self.label6_4 = QtWidgets.QLabel(self)  # Label
        self.label6_4.setText('end epoch:')
        self.label6_4.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_4.move(50, 500)
        self.input6_4 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_4.setText('50')  # default, end_epoch=50
        self.input6_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_4.setGeometry(500, 500, 100, 30)
        self.input6_4.setStyleSheet('''
                                background:#DEDEDE;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')

        # warmup_epoch
        self.label6_5 = QtWidgets.QLabel(self)  # Label
        self.label6_5.setText('warmup epoch:')
        self.label6_5.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_5.move(50, 540)
        self.input6_5 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_5.setText('3')  # default, warmup_epoch=3
        self.input6_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_5.setGeometry(500, 540, 100, 30)
        self.input6_5.setStyleSheet('''
                                        background:#DEDEDE;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')

        # initial_lr
        self.label6_6 = QtWidgets.QLabel(self)  # Label
        self.label6_6.setText('initial learning rate:')
        self.label6_6.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_6.move(50, 580)
        self.input6_6 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_6.setText('0.003')  # default, initial_lr=0.003
        self.input6_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_6.setGeometry(500, 580, 100, 30)
        self.input6_6.setStyleSheet('''
                                background:#DEDEDE;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')

        # lr (Learning rate at the end of warm-up)
        self.label6_7 = QtWidgets.QLabel(self)  # Label
        self.label6_7.setText('learning rate at the end of warm-up:')
        self.label6_7.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_7.move(50, 620)
        self.input6_7 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_7.setText('0.01')  # default, lr=0.01
        self.input6_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_7.setGeometry(500, 620, 100, 30)
        self.input6_7.setStyleSheet('''
                                        background:#DEDEDE;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')

        # minimum_lr (Learning rate from the 50th epoch)
        self.label6_8 = QtWidgets.QLabel(self)  # Label
        self.label6_8.setText('learning rate from the 50th epoch:')
        self.label6_8.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_8.move(50, 660)
        self.input6_8 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_8.setText('0.001')  # default, minimum_lr=0.001
        self.input6_8.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_8.setGeometry(500, 660, 100, 30)
        self.input6_8.setStyleSheet('''
                                    background:#DEDEDE;
                                    color:#1F2B37;
                                    font-size:18px;
                                    font-weight:bold;
                                    font-family:monospace;
                                    ''')

        # batch_size
        self.label6_9 = QtWidgets.QLabel(self)  # Label
        self.label6_9.setText('batch size:')
        self.label6_9.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_9.move(50, 700)
        self.input6_9 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_9.setText('8')  # default, batch_size=8
        self.input6_9.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_9.setGeometry(500, 700, 100, 30)
        self.input6_9.setStyleSheet('''
                                    background:#DEDEDE;
                                    color:#1F2B37;
                                    font-size:18px;
                                    font-weight:bold;
                                    font-family:monospace;
                                    ''')

        # imgsz
        self.label6_10 = QtWidgets.QLabel(self)  # Label
        self.label6_10.setText('resize the image according to this size:')
        self.label6_10.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_10.move(50, 740)
        self.input6_10 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_10.setText('1024')  # default, imgsz=1024
        self.input6_10.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_10.setGeometry(500, 740, 100, 30)
        self.input6_10.setStyleSheet('''
                                            background:#DEDEDE;
                                            color:#1F2B37;
                                            font-size:18px;
                                            font-weight:bold;
                                            font-family:monospace;
                                            ''')

        # use_patch
        self.label6_11 = QtWidgets.QLabel(self)  # Label
        self.label6_11.setText('use patch:')
        self.label6_11.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_11.move(50, 780)
        button_group6_11 = QtWidgets.QButtonGroup(self)
        self.rb_a6_11 = QtWidgets.QRadioButton(self)  # 單選
        button_group6_11.addButton(self.rb_a6_11)
        self.rb_a6_11.setGeometry(400, 780, 100, 20)
        self.rb_a6_11.setText('True')
        self.rb_a6_11.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.rb_b6_11 = QtWidgets.QRadioButton(self)
        button_group6_11.addButton(self.rb_b6_11)
        self.rb_b6_11.setGeometry(530, 780, 100, 20)
        self.rb_b6_11.setText('False')
        self.rb_b6_11.setChecked(True)  # default, use_patch=False
        self.rb_b6_11.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')

        # device
        self.label6_12 = QtWidgets.QLabel(self)  # Label
        self.label6_12.setText('device:')
        self.label6_12.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_12.move(50, 820)
        self.box6_12 = QtWidgets.QComboBox(self)  # 下拉選單
        self.box6_12.addItems(['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3', 'CPU'])
        self.box6_12.setCurrentIndex(0)  # default, Stride=1
        self.box6_12.setGeometry(500, 820, 100, 30)
        self.box6_12.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')

        # Start -> run
        self.run_label = QtWidgets.QLabel(self)  # Label
        self.run_label.setText('Start:')
        self.run_label.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.run_label.move(5, 860)
        self.run_btn1 = QtWidgets.QPushButton(self)  # Button
        self.run_btn1.setGeometry(50, 900, 150, 30)
        self.run_btn1.setText('Training')
        self.run_btn1.setStyleSheet('''
                                QPushButton {
                                    background:#77F2A1;
                                    color:#1F2B37;
                                    font-size:20px;
                                    font-weight:bold;
                                    font-family:monospace;
                                }
                                QPushButton:hover {
                                    background:#1F2B37;
                                    color:#77F2A1;
                                }
                                ''')
        self.run_btn1.clicked.connect(lambda: self.run_program('train'))
        self.run_btn2 = QtWidgets.QPushButton(self)  # Button
        self.run_btn2.setGeometry(250, 900, 150, 30)
        self.run_btn2.setText('Evaluation')
        self.run_btn2.setStyleSheet('''
                                        QPushButton {
                                            background:#77F2A1;
                                            color:#1F2B37;
                                            font-size:20px;
                                            font-weight:bold;
                                            font-family:monospace;
                                        }
                                        QPushButton:hover {
                                            background:#1F2B37;
                                            color:#77F2A1;
                                        }
                                        ''')
        self.run_btn2.clicked.connect(lambda: self.run_program('evaluate'))  # run_program!!!

        # 未完成!!!!
        self.run_btn3 = QtWidgets.QPushButton(self)  # Button
        self.run_btn3.setGeometry(450, 900, 150, 30)
        self.run_btn3.setText('Inference')
        self.run_btn3.setStyleSheet('''
                                        QPushButton {
                                            background:#808080;      
                                            color:#DEDEDE;
                                            font-size:20px;
                                            font-weight:bold;
                                            font-family:monospace;
                                        }
                                        QPushButton:hover {
                                            background:#DEDEDE;
                                            color:#808080;
                                        }
                                        ''')
        self.run_btn3.clicked.connect(lambda: self.run_program('inference'))
        # self.run_btn3.setDisabled(True)     # 暫時禁用

    def openFolder(self, input_line):
        folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        input_line.setText(folderPath)
        # print(folderPath)

    def openFile(self, input_line):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName()
        input_line.setText(filePath)
        # print(filePath)

    def enable_disable_controls(self):  # model -> backbone, training dataset, test dataset
        selected_model = self.box1.currentText()
        if selected_model == 'YOLOv7-Object Detection':
            self.box2.setDisabled(True)
            self.input4.setDisabled(True)
            self.btn4.setDisabled(True)
            self.input5.setDisabled(True)
            self.btn5.setDisabled(True)
            self.input4_2.setDisabled(False)
            self.btn4_2.setDisabled(False)
            self.input5_2.setDisabled(False)
            self.btn5_2.setDisabled(False)
        elif selected_model == 'YOLOv7-Instance Segmentation':
            self.box2.setDisabled(True)
            self.input4.setDisabled(False)
            self.btn4.setDisabled(False)
            self.input5.setDisabled(False)
            self.btn5.setDisabled(False)
            self.input4_2.setDisabled(True)
            self.btn4_2.setDisabled(True)
            self.input5_2.setDisabled(True)
            self.btn5_2.setDisabled(True)
        elif selected_model == 'Cascade-Mask RCNN':
            self.box2.setDisabled(False)
            self.box2.clear()
            self.box2.addItems(['ResNet-50', 'ResNet-101', 'ResNeXt-101'])
            self.input4.setDisabled(True)
            self.btn4.setDisabled(True)
            self.input5.setDisabled(True)
            self.btn5.setDisabled(True)
            self.input4_2.setDisabled(True)
            self.btn4_2.setDisabled(True)
            self.input5_2.setDisabled(True)
            self.btn5_2.setDisabled(True)
        elif selected_model == 'Mask2Former':
            self.box2.setDisabled(False)
            self.box2.clear()
            self.box2.addItems(['ResNet-50', 'ResNet-101', 'Swin Transformer'])
            self.input4.setDisabled(True)
            self.btn4.setDisabled(True)
            self.input5.setDisabled(True)
            self.btn5.setDisabled(True)
            self.input4_2.setDisabled(True)
            self.btn4_2.setDisabled(True)
            self.input5_2.setDisabled(True)
            self.btn5_2.setDisabled(True)

    def modify_cfg(self, task):
        # Model
        model = self.box1.currentText()
        # Backbone
        backbone = self.box2.currentText() if self.box2.currentText() else 'ResNet-50'
        # Paths
        coco_root = self.input3.text()
        train_dir = self.input4.text()
        val_dir = self.input5.text()
        train_txt = self.input4_2.text()
        val_txt = self.input5_2.text()
        cls_path = self.input7.text()
        # Hyperparameter Settings
        optimizer = self.box6_1.currentText()
        weight = self.input6_2.text() if self.input6_2.text() else ''
        start_epoch = self.input6_3.text()
        end_epoch = self.input6_4.text()
        warmup_epoch = self.input6_5.text()
        initial_lr = self.input6_6.text()
        lr = self.input6_7.text()
        minimum_lr = self.input6_8.text()
        batch_size = self.input6_9.text()
        imgsz = self.input6_10.text()
        use_patch = True if self.rb_a6_11.isChecked() else False
        device = self.box6_12.currentText()

        # Check
        model_name_mapping = {'YOLOv7-Object Detection': 'YOLO-v7/yolov7_obj_base',
                              'YOLOv7-Instance Segmentation': 'YOLO-v7/yolov7_inSeg_base',
                              'Cascade-Mask RCNN': 'Cascade-Mask RCNN',
                              'Mask2Former': 'Mask2Former'}  # Cascade & Mask2Former：default backbone=r50
        model_name = model_name_mapping.get(model)
        model_task_mapping = {'YOLOv7-Object Detection': 'object_detection',
                              'YOLOv7-Instance Segmentation': 'instance_segmentation',
                              'Cascade-Mask RCNN': 'instance_segmentation',
                              'Mask2Former': 'instance_segmentation'}
        model_task = model_task_mapping.get(model)
        backbone_name_mapping = {'ResNet-50': 'r50',
                                 'ResNet-101': 'r101',
                                 'ResNeXt-101': 'x101',
                                 'Swin Transformer': 'swin-T'}
        backbone_name = backbone_name_mapping.get(backbone)
        if model_name == 'YOLO-v7/yolov7_obj_base':
            if not all(os.path.exists(path) for path in [train_txt, val_txt, coco_root, cls_path]):
                error_path = "One or more specified paths do not exist."
                QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_path}</font>')
                return
        elif model_name == 'YOLO-v7/yolov7_inSeg_base':
            if not all(os.path.exists(path) for path in [train_dir, val_dir, coco_root, cls_path]):
                error_path = "One or more specified paths do not exist."
                QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_path}</font>')
                return
        else:
            if not all(os.path.exists(path) for path in [coco_root, cls_path]):
                error_path = "One or more specified paths do not exist."
                QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_path}</font>')
                return
        cls_names, cls_num = self.cls_num_type(cls_path)
        if cls_num == 0:
            error_cls = "There is no class in the file"
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_cls}</font>')
            return
        # weight
        if task == 'evaluate' and weight == '':  # inference
            error_weight = "Provide a weight file for evaluation."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_weight}</font>')
            return
        if not self.check_int0(start_epoch, "[start epoch]"):
            return
        if not self.check_int(end_epoch, "[end epoch]"):
            return
        if not self.check_int(warmup_epoch, "[warmup epoch]"):
            return
        if not self.check_float(initial_lr, "[initial learning rate]"):
            return
        if not self.check_float(lr, "[learning rate at the end of warm-up]"):
            return
        if not self.check_float(minimum_lr, "[learning rate from the 50th epoch]"):
            return
        if not self.check_int(batch_size, "[batch size]"):
            return
        if not self.check_int(imgsz, "[resize]"):
            return
        device_mapping = {'GPU 0': '0', 'GPU 1': '1', 'GPU 2': '2', 'GPU 3': '3', 'CPU': 'cpu'}
        device_name = device_mapping.get(device)


        cfg_path = "C:/Users/Yeh/Desktop/AOI-Project/configs/gui_custom.yaml"    ###要改!!!
        with open(cfg_path, 'r', encoding='utf-8') as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        # Model & Backbone
        if model_name == 'Cascade-Mask RCNN' or model_name == 'Mask2Former':
            cfg['_base_'][0] = f"./base/model/{model_name}/{backbone_name}.yaml"
        else:
            cfg['_base_'][0] = f"./base/model/{model_name}.yaml"
        cfg['_base_'][1] = f"./base/evaluation/{model_task}.yaml"
        # Paths
        cfg['coco_root'] = coco_root
        if model_name == 'YOLO-v7/yolov7_obj_base':
            cfg['train_txt'] = train_txt
            cfg['val_txt'] = val_txt
        elif model_name == 'YOLO-v7/yolov7_inSeg_base':
            cfg['train_dir'] = train_dir
            cfg['val_dir'] = val_dir
        cfg['number_of_class'] = cls_num
        cfg['class_names'] = cls_names
        # Hyperparameter
        cfg['optimizer'] = optimizer
        cfg['weight'] = weight
        cfg['start_epoch'] = int(start_epoch)
        cfg['end_epoch'] = int(end_epoch)
        cfg['warmup_epoch'] = int(warmup_epoch)
        cfg['initial_lr'] = float(initial_lr)
        cfg['lr'] = float(lr)
        cfg['minimum_lr'] = float(minimum_lr)
        cfg['batch_size'] = int(batch_size)
        cfg['imgsz'] = [int(imgsz), int(imgsz)]
        cfg['use_patch'] = use_patch
        cfg['device'] = device_name

        with open(cfg_path, 'w', encoding='utf-8') as cfg_file:
            yaml.dump(cfg, cfg_file)

    def run_program(self, task):
        task_mapping = {'train': 'train.py',
                        'evaluate': 'evaluation.py'}  # inference
        task_py = task_mapping.get(task)

        self.modify_cfg(task)

        cmd = [
            "python",
            f"C:/Users/Yeh/Desktop/AOI-Project/tools/{task_py}",  ###要改!!!
            "-c", "./configs/gui_custom.yaml"    ###要改!!!
        ]
        # Run
        subprocess.run(cmd, check=True)

    def check_int(self, input_text, error_message):  # >0
        if not input_text.isdigit() or int(input_text) <= 0:
            error_int = f"{error_message} must be a positive integer."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_int}</font>')
            return False
        return True

    def check_int0(self, input_text, error_message):  # >=0
        if not input_text.isdigit() or int(input_text) < 0:
            error_int0 = f"{error_message} Must be an integer greater than or equal to 0."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_int0}</font>')
            return False
        return True

    def check_float(self, input_text, error_message):  # 0-1
        if not self.is_float(input_text) or float(input_text) < 0 or float(input_text) > 1:
            error_float = f"{error_message} must be a number in the range 0 to 1."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_float}</font>')
            return False
        return True

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def cls_num_type(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file)
                if not content:
                    return None, 0

                cls_info_list = []
                for class_info in content.values():
                    super_cls = class_info.get('super')
                    if super_cls is not None:
                        cls_info_list.append((super_cls, class_info.get('id', float('inf'))))
                if not cls_info_list:
                    return None, 0

                unique_super_cls_set = set()
                sorted_cls_info_list = sorted(cls_info_list, key=lambda x: x[1])
                unique_super_cls = [cls_info[0] for cls_info in sorted_cls_info_list if
                                    cls_info[0] not in unique_super_cls_set
                                    and not unique_super_cls_set.add(cls_info[0])]
                return unique_super_cls, len(unique_super_cls)

        except Exception as e:
            error_weight = "Not a yaml file."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_weight}</font>')
            return None, 0


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec())
