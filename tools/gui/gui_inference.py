from PyQt6 import QtWidgets, QtCore
import sys
import os
import subprocess
import yaml


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI-Inference')
        self.resize(900, 800)
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
        self.box1.currentTextChanged.connect(self.controls_backbone)  # 控制 backbone

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

        # classes_yaml
        self.label7 = QtWidgets.QLabel(self)  # Label
        self.label7.setText('Path to classes.yaml file:')
        self.label7.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label7.move(5, 100)
        self.input7 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input7.setGeometry(300, 100, 300, 30)
        self.input7.setStyleSheet('''
                                        background:#DEDEDE;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.btn7 = QtWidgets.QPushButton(self)  # Button
        self.btn7.move(650, 100)
        self.btn7.setText('Browse')
        self.btn7.setStyleSheet('''
                                        background:#77F2A1;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.btn7.clicked.connect(lambda: self.openFile(self.input7))

        # weight
        self.label6_2 = QtWidgets.QLabel(self)  # Label
        self.label6_2.setText('Weight file path:')
        self.label6_2.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_2.move(5, 140)
        self.input6_2 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_2.setGeometry(300, 140, 300, 30)
        self.input6_2.setStyleSheet('''
                                            background:#DEDEDE;
                                            color:#1F2B37;
                                            font-size:18px;
                                            font-weight:bold;
                                            font-family:monospace;
                                            ''')
        self.btn6_2 = QtWidgets.QPushButton(self)  # Button
        self.btn6_2.move(650, 140)
        self.btn6_2.setText('Browse')
        self.btn6_2.setStyleSheet('''
                                        background:#77F2A1;
                                        color:#1F2B37;
                                        font-size:18px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.btn6_2.clicked.connect(lambda: self.openFile(self.input6_2))

        # imgsz
        self.label6_10 = QtWidgets.QLabel(self)  # Label
        self.label6_10.setText('Resize the image according to this size:')
        self.label6_10.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6_10.move(5, 180)
        self.input6_10 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6_10.setText('1024')  # default, imgsz=1024
        self.input6_10.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input6_10.setGeometry(500, 180, 100, 30)
        self.input6_10.setStyleSheet('''
                                            background:#DEDEDE;
                                            color:#1F2B37;
                                            font-size:18px;
                                            font-weight:bold;
                                            font-family:monospace;
                                            ''')

        # Folder path to predict
        self.label8 = QtWidgets.QLabel(self)  # Label
        self.label8.setText('Folder path to predict:')
        self.label8.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label8.move(5, 220)
        self.input8 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input8.setGeometry(300, 220, 300, 30)
        self.input8.setStyleSheet('''
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
        self.btn8 = QtWidgets.QPushButton(self)  # Button
        self.btn8.move(650, 220)
        self.btn8.setText('Browse')
        self.btn8.setStyleSheet('''
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
        self.btn8.clicked.connect(lambda: self.openFolder(self.input8))
        self.input8.textChanged.connect(self.controls_predict)
        # Image path to predict
        self.label8_2 = QtWidgets.QLabel(self)  # Label
        self.label8_2.setText('Image path to predict:')
        self.label8_2.setStyleSheet('''
                                                color:#DEDEDE;
                                                font-size:20px;
                                                font-weight:bold;
                                                font-family:monospace;
                                                ''')
        self.label8_2.move(5, 260)
        self.input8_2 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input8_2.setGeometry(300, 260, 300, 30)
        self.input8_2.setStyleSheet('''
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
        self.btn8_2 = QtWidgets.QPushButton(self)  # Button
        self.btn8_2.move(650, 260)
        self.btn8_2.setText('Browse')
        self.btn8_2.setStyleSheet('''
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
        self.btn8_2.clicked.connect(lambda: self.openFile(self.input8_2))
        self.input8_2.textChanged.connect(self.controls_predict)

        # confidence
        self.label9 = QtWidgets.QLabel(self)  # Label
        self.label9.setText('Confidence threshold (Optional):')
        self.label9.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.label9.move(5, 300)
        self.input9 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input9.setText('0.5')  # default, confidence_threshold=0.5
        self.input9.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input9.setGeometry(500, 300, 100, 30)
        self.input9.setStyleSheet('''
                                                    background:#DEDEDE;
                                                    color:#1F2B37;
                                                    font-size:18px;
                                                    font-weight:bold;
                                                    font-family:monospace;
                                                    ''')

        # nms
        self.label9_2 = QtWidgets.QLabel(self)  # Label
        self.label9_2.setText('NMS threshold (Optional):')
        self.label9_2.setStyleSheet('''
                                                color:#DEDEDE;
                                                font-size:20px;
                                                font-weight:bold;
                                                font-family:monospace;
                                                ''')
        self.label9_2.move(5, 340)
        self.input9_2 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input9_2.setText('0.5')  # default, nms_threshold=0.5
        self.input9_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        self.input9_2.setGeometry(500, 340, 100, 30)
        self.input9_2.setStyleSheet('''
                                                            background:#DEDEDE;
                                                            color:#1F2B37;
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
        self.run_label.move(5, 380)
        self.run_label = QtWidgets.QLabel(self)  # Label
        self.run_label.setText('(The prediction results will be stored in "AOI-Project/work_dirs/predict")')
        self.run_label.setStyleSheet('''
                                                color:#DEDEDE;
                                                font-size:18px;
                                                font-weight:bold;
                                                font-family:monospace;
                                                ''')
        self.run_label.move(5, 420)
        self.run_btn1 = QtWidgets.QPushButton(self)  # Button
        self.run_btn1.setGeometry(250, 460, 150, 30)
        self.run_btn1.setText('Inference')
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
        self.run_btn1.clicked.connect(self.run_program)


    def openFolder(self, input_line):
        folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        input_line.setText(folderPath)
        # print(folderPath)

    def openFile(self, input_line):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName()
        input_line.setText(filePath)
        # print(filePath)

    def controls_backbone(self):  # model -> backbone
        selected_model = self.box1.currentText()
        if selected_model == 'YOLOv7-Object Detection':
            self.box2.setDisabled(True)
        elif selected_model == 'YOLOv7-Instance Segmentation':
            self.box2.setDisabled(True)
        elif selected_model == 'Mask2Former':
            self.box2.setDisabled(False)
            self.box2.clear()
            self.box2.addItems(['ResNet-50', 'ResNet-101', 'Swin Transformer'])

    def controls_predict(self):  # predict -> folder or image
        folder_text = self.input8.text()
        image_text = self.input8_2.text()
        # Choose one
        if folder_text:
            self.input8_2.setDisabled(True)
            self.btn8_2.setDisabled(True)
        elif image_text:
            self.input8.setDisabled(True)
            self.btn8.setDisabled(True)
        else:   # If both are empty, enable both
            self.input8.setDisabled(False)
            self.btn8.setDisabled(False)
            self.input8_2.setDisabled(False)
            self.btn8_2.setDisabled(False)

    def modify_cfg(self):
        # ==========parameter==========
        model = self.box1.currentText()
        backbone = self.box2.currentText() if self.box2.currentText() else 'ResNet-50'
        cls_path = self.input7.text()
        weight_path = self.input6_2.text()
        imgsz = self.input6_10.text()
        predict_path = self.input8.text() if self.input8.text() else self.input8_2.text()
        confidence = self.input9.text()
        nms = self.input9_2.text()

        # ==========Check==========
        # model, backbone
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
        # cls_path, weight_path, predict_path
        if not all(os.path.exists(path) for path in [cls_path, weight_path, predict_path]):
            error_path = "One or more specified paths do not exist."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_path}</font>')
            return
        # cls_path
        cls_names, cls_num = self.cls_num_type(cls_path)
        if cls_num == 0:
            error_cls = "There is no class in the file"
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_cls}</font>')
            return
        # imgsz
        if not self.check_int(imgsz, "[resize]"):
            return
        # confidence
        if not self.check_float(confidence, "[confidence threshold]"):
            return
        # nms
        if not self.check_float(nms, "[NMS]"):
            return


        # ==========Modify==========
        # read yaml
        cfg_path = "./configs/gui_custom.yaml"    ###要改!!!
        with open(cfg_path, 'r', encoding='utf-8') as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        # modify
        if model_name == 'Cascade-Mask RCNN' or model_name == 'Mask2Former':
            cfg['_base_'][0] = f"./base/model/{model_name}/{backbone_name}.yaml"
        else:
            cfg['_base_'][0] = f"./base/model/{model_name}.yaml"
        cfg['_base_'][1] = f"./base/evaluation/{model_task}.yaml"
        cfg['number_of_class'] = cls_num
        cfg['class_names'] = cls_names
        cfg['weight'] = weight_path
        cfg['imgsz'] = [int(imgsz), int(imgsz)]
        # save yaml
        with open(cfg_path, 'w', encoding='utf-8') as cfg_file:
            yaml.dump(cfg, cfg_file)


    def run_program(self):
        # parameter
        predict_path = self.input8.text() if self.input8.text() else self.input8_2.text()
        confidence = self.input9.text()
        nms = self.input9_2.text()

        # modify
        self.modify_cfg()
        cmd = ["python", "./tools/predict.py",
               "--config", "./configs/gui_custom.yaml",
               "--source", f"{predict_path}",
               "--conf_thres", f"{confidence}",
               "--nms_thres", f"{nms}"]

        # Run
        try:
            subprocess.run(cmd, check=False)
        except subprocess.CalledProcessError as e:
            print(f'Error: Command returned non-zero exit status. {e}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')

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
