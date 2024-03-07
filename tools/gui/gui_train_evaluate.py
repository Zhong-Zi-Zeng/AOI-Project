from PyQt6 import QtWidgets, QtCore
import sys
import subprocess
import yaml



class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI-Training & Evaluation')
        self.resize(900, 1000)
        self.setStyleSheet('background:#1F2B37')
        self.ui()

    def drop_down_list1(self, name, y, setText, addItems, default=None, control_func=None):
        label = QtWidgets.QLabel(self)  # Label
        label.setText(setText)
        label.setStyleSheet('''
                                                color:#DEDEDE;
                                                font-size:20px;
                                                font-weight:bold;
                                                font-family:monospace;
                                                ''')
        label.move(5, y)
        box = QtWidgets.QComboBox(self)  # 下拉選單
        box.addItems(addItems)
        if default:
            box.setCurrentIndex(default)  # default
        box.setGeometry(300, y, 300, 30)
        box.setStyleSheet('''
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
        box.view().setStyleSheet('''  
                                    QAbstractItemView {
                                        color: #DEDEDE;
                                        background-color: #1F2B37;
                                    }
                                ''')  # 解決QComboBox時，無法實現 color: #DEDEDE
        if control_func:
            box.currentTextChanged.connect(control_func)

        setattr(self, f'label_{name}', label)
        setattr(self, f'box_{name}', box)

    def drop_down_list2(self, name, y, setText, addItems, x1=50, x2=500, default=None):
        label = QtWidgets.QLabel(self)  # Label
        label.setText(setText)
        label.setStyleSheet('''
                            color:#DEDEDE;
                            font-size:18px;
                            font-weight:bold;
                            font-family:monospace;
                            ''')
        label.move(x1, y)
        box = QtWidgets.QComboBox(self)  # 下拉選單
        box.addItems(addItems)
        box.setCurrentIndex(0)  # default
        box.setGeometry(x2, y, 100, 30)
        box.setStyleSheet('''
                            color:#DEDEDE;
                            font-size:18px;
                            font-weight:bold;
                            font-family:monospace;
                            ''')

        setattr(self, f'label_{name}', label)
        setattr(self, f'box_{name}', box)

    def input_box1(self, name, y, setText, label_position=5, x1=300, x2=650, default=None, font_size=20, input_size=300, File=False):
        label = QtWidgets.QLabel(self)  # Label
        label.setText(setText)
        label.setStyleSheet(f'''
                            color:#DEDEDE;
                            font-size:{font_size}px;
                            font-weight:bold;
                            font-family:monospace;
                            ''')
        label.move(label_position, y)
        input = QtWidgets.QLineEdit(self)  # 單行輸入框
        input.setGeometry(x1, y, input_size, 30)
        input.setStyleSheet('''
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
        input.setDisabled(default)  # default
        btn = QtWidgets.QPushButton(self)  # Button
        btn.move(x2, y)
        btn.setText('Browse')
        btn.setStyleSheet('''
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
        if File:
            btn.clicked.connect(lambda: self.openFile(input))
        else:
            btn.clicked.connect(lambda: self.openFolder(input))
        btn.setDisabled(default)  # default

        setattr(self, f'label_{name}', label)
        setattr(self, f'input_{name}', input)
        setattr(self, f'btn_{name}', btn)

    def input_box2(self, name, x1, x2, y, setText, default):
        label = QtWidgets.QLabel(self)  # Label
        label.setText(setText)
        label.setStyleSheet('''
                            color:#DEDEDE;
                            font-size:18px;
                            font-weight:bold;
                            font-family:monospace;
                            ''')
        label.move(x1, y)
        input = QtWidgets.QLineEdit(self)  # 單行輸入框
        input.setText(default)
        input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        input.setGeometry(x2, y, 100, 30)
        input.setStyleSheet('''
                                        background:#DEDEDE;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')

        setattr(self, f'label_{name}', label)
        setattr(self, f'input_{name}', input)


    def ui(self):
        # ====================config file====================
        # task
        addItems = ['Image Segmentation', 'Object Detection']
        self.drop_down_list1('task', 20, 'Task used:', addItems, default=1, control_func=self.control_task)  # 下拉選單
            # default, task=Object Detection
            # 控制 model

        # model
        addItems = ['YOLOv7', 'YOLOv7-X', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E', 'CO-DETR','EfficientDet']
            # 所有model種類在control_task
        self.drop_down_list1('model', 60, 'Model used:', addItems, default=0, control_func=self.control_model)  # 下拉選單
            # default, model=YOLOv7
            # 控制 backbone, training dataset, test dataset

        # backbone
        addItems = ['ResNet-50']    # 所有backbone種類在control_model
        self.drop_down_list1('backbone', 100, 'Backbone used:', addItems)  # 下拉選單

        # ====================================
        # coco_root
        self.input_box1('coco_root', 140, 'Path to coco dataset:', default=False)

        # training dataset：folder -> yolov7-inSeg
        self.input_box1('train_dataset_folder', 180, 'Path to training folder:', default=True)
            # default：yolov7-obj

        # test dataset：folder -> yolov7-inSeg
        self.input_box1('test_dataset_folder', 220, 'Path to test folder:', default=True)
            # default：yolov7-obj

        # training dataset：txt -> yolov7-obj
        self.input_box1('train_dataset_txt', 260, 'Path to train txt file:', default=False, File=True)
            # default：yolov7-obj

        # test dataset：txt -> yolov7-obj
        self.input_box1('test_dataset_txt', 300, 'Path to test txt file:', default=False, File=True)
            # default：yolov7-obj

        # =========================================
        # Augmentation
        self.label_hyp = QtWidgets.QLabel(self)  # Label
        self.label_hyp.setText('Augmentation:')
        self.label_hyp.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label_hyp.move(5, 380)

        # HSV-Hue
        self.input_box2('hsv_h', 50, 200, 420, 'hsv_h:', '0.15')
            # default, hsv_h=0.15

        # HSV-Saturation
        self.input_box2('hsv_s', 50, 200, 460, 'hsv_s:', '0.7')
            # default, hsv_s=0.7

        # HSV-Value
        self.input_box2('hsv_v', 50, 200, 500, 'hsv_v:', '0.8')
            # default, hsv_v=0.8

        # rotation
        self.input_box2('degrees', 50, 200, 540, 'degrees:', '0.')
            # default, degrees=0.

        # translation
        self.input_box2('translate', 50, 200, 580, 'translate:', '0.2')
            # default, translate=0.2

        # scale
        self.input_box2('scale', 50, 200, 620, 'scale:', '0.9')
            # default, scale=0.9

        # shear
        self.input_box2('shear', 50, 200, 660, 'shear:', '0.')
            # default, shear=0.

        # perspective
        self.input_box2('perspective', 50, 200, 700, 'perspective:', '0.')
            # default, perspective=0.

        # flip up-down
        self.input_box2('flipud', 50, 200, 740, 'flipud:', '0.')
            # default, flipud=0.

        # flip left-right
        self.input_box2('fliplr', 50, 200, 780, 'fliplr:', '0.5')
            # default, fliplr=0.5

        # mosaic
        self.input_box2('mosaic', 50, 200, 820, 'mosaic:', '1.')
            # default, mosaic=1.

        # mixup
        self.input_box2('mixup', 50, 200, 860, 'mixup:', '0.15')
            # default, mixup=0.15

        # copy-paste
        self.input_box2('copy_paste', 50, 200, 900, 'copy_paste:', '0.0')
            # default, copy_paste=0.0

        # =========================================
        # Hyperparameter
        self.label_hyp = QtWidgets.QLabel(self)  # Label
        self.label_hyp.setText('Hyperparameter:')
        self.label_hyp.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label_hyp.move(400, 380)

        # optimizer
        addItems = ['Adam', 'AdamW', 'SGD']
        self.drop_down_list2('optimizer', 420, 'optimizer:', addItems, x1=450, x2=620, default=2)  # 下拉選單
            # default, optimizer=SGD

        # weight
        self.input_box1('weight', 460, 'weight file path:', label_position=450, x1=620, x2=750, default=False, font_size=18, input_size=100, File=True)

        # start_epoch
        self.input_box2('start_epoch', 450, 620, 500, 'start epoch:', '0')
            # default, start_epoch=0

        # end_epoch
        self.input_box2('end_epoch', 450, 620, 540, 'end epoch:', '50')
            # default, end_epoch=50

        # warmup_epoch
        self.input_box2('warmup_epoch', 450, 620, 580, 'warmup epoch:', '3')
            # default, warmup_epoch=3

        # initial_lr
        self.input_box2('initial_lr', 450, 620, 620, 'initial lr:', '0.003')
            # default, initial_lr=0.003

        # lr (Learning rate at the end of warm-up)
        self.input_box2('lr', 450, 620, 660, 'lr:', '0.01')
            # default, lr=0.01

        # minimum_lr (Learning rate from the 50th epoch)
        self.input_box2('minimum_lr', 450, 620, 700, 'minimum lr:', '0.01')
            # default, minimum_lr=0.001

        # batch_size
        self.input_box2('batch_size', 450, 620, 740, 'batch size:', '8')
            # default, batch_size=8

        # imgsz
        self.input_box2('imgsz', 450, 620, 780, 'resize:', '1024')
            # default, imgsz=1024

        # device
        addItems = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3', 'CPU']
        self.drop_down_list2('device', 820, 'device:', addItems, x1=450, x2=620, default=0)  # 下拉選單
            # default, device=0

        # Start -> run
        self.run_label = QtWidgets.QLabel(self)  # Label
        self.run_label.setText('Start:')
        self.run_label.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.run_label.move(5, 940)
        self.run_btn1 = QtWidgets.QPushButton(self)  # Button
        self.run_btn1.setGeometry(150, 960, 150, 30)
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
        self.run_btn2.setGeometry(450, 960, 150, 30)
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
        self.run_btn2.clicked.connect(lambda: self.run_program('evaluate'))

    def openFolder(self, input_line):
        folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        input_line.setText(folderPath)
        # print(folderPath)

    def openFile(self, input_line):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName()
        input_line.setText(filePath)
        # print(filePath)
    def control_task(self):
        Obj_list = ['YOLOv7', 'YOLOv7-X', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E', 'CO-DETR', 'EfficientDet']
        Seg_list = ['YOLOv7 Instance Segmentation', 'Segment Anything', 'Cascade Mask RCNN', 'Mask2Former']

        selected_task = self.box_task.currentText()
        if selected_task == 'Image Segmentation':
            self.box_model.setDisabled(False)
            self.box_model.clear()
            self.box_model.addItems(Seg_list)
        elif selected_task == 'Object Detection':
            self.box_model.setDisabled(False)
            self.box_model.clear()
            self.box_model.addItems(Obj_list)

    def control_model(self):  # model -> backbone, training dataset, test dataset
        model_no_backbone = ['YOLOv7', 'YOLOv7-X', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E', 'YOLOv7 Instance Segmentation']
        dataset_txt = ['YOLOv7', 'YOLOv7-X', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E']

        selected_model = self.box_model.currentText()
        if selected_model in model_no_backbone:
            self.box_backbone.setDisabled(True)

            if selected_model in dataset_txt: # txt
                self.input_train_dataset_folder.setDisabled(True)
                self.btn_train_dataset_folder.setDisabled(True)
                self.input_test_dataset_folder.setDisabled(True)
                self.btn_test_dataset_folder.setDisabled(True)
                self.input_train_dataset_txt.setDisabled(False)
                self.btn_train_dataset_txt.setDisabled(False)
                self.input_test_dataset_txt.setDisabled(False)
                self.btn_test_dataset_txt.setDisabled(False)
            elif selected_model == 'YOLOv7 Instance Segmentation':  # folder
                self.input_train_dataset_folder.setDisabled(False)
                self.btn_train_dataset_folder.setDisabled(False)
                self.input_test_dataset_folder.setDisabled(False)
                self.btn_test_dataset_folder.setDisabled(False)
                self.input_train_dataset_txt.setDisabled(True)
                self.btn_train_dataset_txt.setDisabled(True)
                self.input_test_dataset_txt.setDisabled(True)
                self.btn_test_dataset_txt.setDisabled(True)
            else:
                self.input_train_dataset_folder.setDisabled(True)
                self.btn_train_dataset_folder.setDisabled(True)
                self.input_test_dataset_folder.setDisabled(True)
                self.btn_test_dataset_folder.setDisabled(True)
                self.input_train_dataset_txt.setDisabled(True)
                self.btn_train_dataset_txt.setDisabled(True)
                self.input_test_dataset_txt.setDisabled(True)
                self.btn_test_dataset_txt.setDisabled(True)
        elif selected_model == 'CO-DETR':
            self.box_backbone.setDisabled(False)
            self.box_backbone.clear()
            self.box_backbone.addItems(['ResNet-50'])

            self.input_train_dataset_folder.setDisabled(True)
            self.btn_train_dataset_folder.setDisabled(True)
            self.input_test_dataset_folder.setDisabled(True)
            self.btn_test_dataset_folder.setDisabled(True)
            self.input_train_dataset_txt.setDisabled(True)
            self.btn_train_dataset_txt.setDisabled(True)
            self.input_test_dataset_txt.setDisabled(True)
            self.btn_test_dataset_txt.setDisabled(True)
        elif selected_model == 'EfficientDet':
            self.box_backbone.setDisabled(False)
            self.box_backbone.clear()
            self.box_backbone.addItems(['EfficientNet-d0', 'EfficientNet-d3'])

            self.input_train_dataset_folder.setDisabled(True)
            self.btn_train_dataset_folder.setDisabled(True)
            self.input_test_dataset_folder.setDisabled(True)
            self.btn_test_dataset_folder.setDisabled(True)
            self.input_train_dataset_txt.setDisabled(True)
            self.btn_train_dataset_txt.setDisabled(True)
            self.input_test_dataset_txt.setDisabled(True)
            self.btn_test_dataset_txt.setDisabled(True)
        elif selected_model == 'Segment Anything':
            self.box_backbone.setDisabled(False)
            self.box_backbone.clear()
            self.box_backbone.addItems(['vit-b', 'vit-l', 'vit-h'])

            self.input_train_dataset_folder.setDisabled(True)
            self.btn_train_dataset_folder.setDisabled(True)
            self.input_test_dataset_folder.setDisabled(True)
            self.btn_test_dataset_folder.setDisabled(True)
            self.input_train_dataset_txt.setDisabled(True)
            self.btn_train_dataset_txt.setDisabled(True)
            self.input_test_dataset_txt.setDisabled(True)
            self.btn_test_dataset_txt.setDisabled(True)
        elif selected_model == 'Cascade Mask RCNN':
            self.box_backbone.setDisabled(False)
            self.box_backbone.clear()
            self.box_backbone.addItems(['ResNet-50', 'ResNet-101', 'ResNeXt-101'])

            self.input_train_dataset_folder.setDisabled(True)
            self.btn_train_dataset_folder.setDisabled(True)
            self.input_test_dataset_folder.setDisabled(True)
            self.btn_test_dataset_folder.setDisabled(True)
            self.input_train_dataset_txt.setDisabled(True)
            self.btn_train_dataset_txt.setDisabled(True)
            self.input_test_dataset_txt.setDisabled(True)
            self.btn_test_dataset_txt.setDisabled(True)
        elif selected_model == 'Mask2Former':
            self.box_backbone.setDisabled(False)
            self.box_backbone.clear()
            self.box_backbone.addItems(['ResNet-50', 'ResNet-101', 'Swin-T'])

            self.input_train_dataset_folder.setDisabled(True)
            self.btn_train_dataset_folder.setDisabled(True)
            self.input_test_dataset_folder.setDisabled(True)
            self.btn_test_dataset_folder.setDisabled(True)
            self.input_train_dataset_txt.setDisabled(True)
            self.btn_train_dataset_txt.setDisabled(True)
            self.input_test_dataset_txt.setDisabled(True)
            self.btn_test_dataset_txt.setDisabled(True)

    # def modify_cfg(self, train_or_evaluate):
    #     # Mapping
    #     # task
    #     task_mapping = {'Object Detection': 'object_detection'}
    #     task_name = task_mapping.get(self.box_task.currentText())
    #     # model
    #     model_name_mapping = {'YOLOv7': 'YOLO-v7/yolov7_obj_base'}
    #     model_name = model_name_mapping.get(self.box_model.currentText())
    #     # backbone
    #     # device
    #     device_mapping = {'GPU 0': '0'}
    #     device_name = device_mapping.get(self.box_device.currentText())
    #
    #     # Check
    #         # weight
    #     if train_or_evaluate == 'evaluate' and weight == '':
    #         error_weight = "Provide a weight file for evaluation."
    #         QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_weight}</font>')
    #         return
    #
    #     # Write
    #     cfg_path = "./configs/gui_custom.yaml"
    #     with open(cfg_path, 'r', encoding='utf-8') as cfg_file:
    #         cfg = yaml.safe_load(cfg_file)
    #         # path
    #     cfg['coco_root'] = self.input_coco_root.text()
    #     cfg['train_txt'] = self.input_train_dataset_txt.text()
    #     cfg['val_txt'] = self.input_test_dataset_txt.text()
    #         # aug
    #     cfg['hsv_h'] = float(self.input_hsv_h.text())
    #     cfg['hsv_s'] = float(self.input_hsv_s.text())
    #     cfg['hsv_v'] = float(self.input_hsv_v.text())
    #     cfg['degrees'] = float(self.input_degrees.text())
    #     cfg['translate'] = float(self.input_translate.text())
    #     cfg['scale'] = float(self.input_scale.text())
    #     cfg['shear'] = float(self.input_shear.text())
    #     cfg['perspective'] = float(self.input_perspective.text())
    #     cfg['flipud'] = float(self.input_flipud.text())
    #     cfg['fliplr'] = float(self.input_fliplr.text())
    #     cfg['mosaic'] = float(self.input_mosaic.text())
    #     cfg['mixup'] = float(self.input_mixup.text())
    #     cfg['copy_paste'] = float(self.input_copy_paste.text())
    #         # hyp
    #     cfg['optimizer'] = self.box_optimizer.currentText()
    #     cfg['weight'] = self.input_weight.text() if self.input_weight.text() else ''
    #     cfg['start_epoch'] = int(self.input_start_epoch.text())
    #     cfg['end_epoch'] = int(self.input_end_epoch.text())
    #     cfg['warmup_epoch'] = int(self.input_warmup_epoch.text())
    #     cfg['initial_lr'] = float(self.input_initial_lr.text())
    #     cfg['lr'] = float(self.input_lr.text())
    #     cfg['minimum_lr'] = float(self.input_minimum_lr.text())
    #     cfg['batch_size'] = int(self.input_batch_size.text())
    #     cfg['imgsz'] = [int(self.input_imgsz.text()), int(self.input_imgsz.text())]
    #     cfg['device'] = device_name
    #
    #     with open(cfg_path, 'w', encoding='utf-8') as cfg_file:
    #         yaml.dump(cfg, cfg_file)

    def run_program(self, train_or_evaluate):
        task_mapping = {'train': 'train.py',
                        'evaluate': 'evaluation.py'}
        task_py = task_mapping.get(train_or_evaluate)

        # self.modify_cfg(train_or_evaluate)

        cmd = [
            "python", f"./tools/{task_py}",
            "-c", "./configs/gui_custom.yaml"]
        # Run
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec())
