from PyQt6 import QtWidgets, QtCore
import sys
import os
import subprocess
import yaml


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI-Inference')
        self.resize(800, 500)
        self.setStyleSheet('background:#1F2B37')
        self.ui()

    def input_box1(self, name, y, setText, Folder=False, control_func=None):
        label = QtWidgets.QLabel(self)  # Label
        label.setText(setText)
        label.setStyleSheet('''
                            color:#DEDEDE;
                            font-size:20px;
                            font-weight:bold;
                            font-family:monospace;
                            ''')
        label.move(5, y)
        input = QtWidgets.QLineEdit(self)  # 單行輸入框
        input.setGeometry(300, y, 300, 30)
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
        btn = QtWidgets.QPushButton(self)  # Button
        btn.move(650, y)
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
        if Folder:
            btn.clicked.connect(lambda: self.openFolder(input))
        else:
            btn.clicked.connect(lambda: self.openFile(input))

        if control_func:
            input.textChanged.connect(control_func)

        setattr(self, f'label_{name}', label)
        setattr(self, f'input_{name}', input)
        setattr(self, f'btn_{name}', btn)

    def input_box2(self, name, y, setText, default):
        label = QtWidgets.QLabel(self)  # Label
        label.setText(setText)
        label.setStyleSheet('''
                            color:#DEDEDE;
                            font-size:20px;
                            font-weight:bold;
                            font-family:monospace;
                            ''')
        label.move(5, y)
        input = QtWidgets.QLineEdit(self)  # 單行輸入框
        input.setText(default)
        input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)  # Align right
        input.setGeometry(500, y, 100, 30)
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
        # final_config
        self.input_box1('config_path', 20, 'Path to config file:')

        # weight
        self.input_box1('weight_path', 60, 'Path to weight file:')

        # Folder path to predict
        self.input_box1('predict_folder', 100, 'Folder path to predict:', Folder=True, control_func=self.control_predict)

        # Image path to predict
        self.input_box1('predict_image', 140, 'Image path to predict:', control_func=self.control_predict)

        # confidence
        self.input_box2('confidence', 180, 'Confidence threshold (Optional):', '0.5')
            # default, confidence_threshold=0.5

        # nms
        self.input_box2('nms', 220, 'NMS threshold (Optional):', '0.5')
            # default, nms_threshold=0.5

        # Start -> run
        self.run_label = QtWidgets.QLabel(self)  # Label
        self.run_label.setText('Start:')
        self.run_label.setStyleSheet('''
                                        color:#DEDEDE;
                                        font-size:20px;
                                        font-weight:bold;
                                        font-family:monospace;
                                        ''')
        self.run_label.move(5, 260)
        self.run_label = QtWidgets.QLabel(self)  # Label
        self.run_label.setText('(The prediction results will be stored in "AOI-Project/work_dirs/predict")')
        self.run_label.setStyleSheet('''
                                                color:#DEDEDE;
                                                font-size:18px;
                                                font-weight:bold;
                                                font-family:monospace;
                                                ''')
        self.run_label.move(5, 300)
        self.run_btn1 = QtWidgets.QPushButton(self)  # Button
        self.run_btn1.setGeometry(250, 340, 150, 30)
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

    def control_predict(self):  # predict -> folder or image
        # Choose one
        if self.input_predict_folder.text():
            self.input_predict_image.setDisabled(True)
            self.btn8_predict_image.setDisabled(True)
        elif self.input_predict_image.text():
            self.input_predict_folder.setDisabled(True)
            self.btn_predict_folder.setDisabled(True)
        else:   # If both are empty, enable both
            self.input_predict_folder.setDisabled(False)
            self.btn_predict_folder.setDisabled(False)
            self.input_predict_image.setDisabled(False)
            self.btn_predict_image.setDisabled(False)

    def modify_cfg(self):
        cfg_path = self.input_config_path.text()
        with open(cfg_path, 'r', encoding='utf-8') as cfg_file:
            cfg = yaml.safe_load(cfg_file)

        cfg['weight'] = self.input_weight_path.text()
        cfg['conf_thres'] = self.input_confidence.text()
        cfg['nms_thres'] = self.input_nms.text()

        with open(cfg_path, 'w', encoding='utf-8') as cfg_file:
            yaml.dump(cfg, cfg_file)


    def run_program(self):
        predict_path = self.input_predict_folder.text() if self.input_predict_folder.text() else self.input_predict_image.text()

        self.modify_cfg()
        cmd = ["python", "./tools/predict.py",
               "--config", self.input_config_path.text(),
               "--source", f"{predict_path}",
               '--show']

        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec())
