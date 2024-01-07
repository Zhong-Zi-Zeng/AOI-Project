from PyQt6 import QtWidgets
import sys
import os
import subprocess
from PIL import Image


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI-Label Converter')
        self.resize(900, 600)
        self.setStyleSheet('background:#1F2B37')
        self.ui()


    def ui(self):
        # ====================create data====================
        # source_dir
        self.label1 = QtWidgets.QLabel(self)    # Label
        self.label1.setText('Path to the original train or test dataset:')
        self.label1.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label1.move(5, 20)
        self.input1 = QtWidgets.QLineEdit(self)     # 單行輸入框
        self.input1.setGeometry(440, 20, 250, 30)
        self.input1.setStyleSheet('''
                                background:#DEDEDE;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.btn1 = QtWidgets.QPushButton(self)     # Button
        self.btn1.move(720, 20)
        self.btn1.setText('Browse')
        self.btn1.setStyleSheet('''
                                background:#77F2A1;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.btn1.clicked.connect(lambda: self.openFolder(self.input1))

        # output_dir
        self.label2 = QtWidgets.QLabel(self)  # Label
        self.label2.setText('Path to the output folder:')
        self.label2.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label2.move(5, 60)
        self.input2 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input2.setGeometry(300, 60, 250, 30)
        self.input2.setStyleSheet('''
                                background:#DEDEDE;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.btn2 = QtWidgets.QPushButton(self)  # Button
        self.btn2.move(580, 60)
        self.btn2.setText('Browse')
        self.btn2.setStyleSheet('''
                                background:#77F2A1;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.btn2.clicked.connect(lambda: self.openFolder(self.input2))

        # classes_yaml
        self.label3 = QtWidgets.QLabel(self)  # Label
        self.label3.setText('Path to classes.yaml file:')
        self.label3.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label3.move(5, 100)
        self.input3 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input3.setGeometry(300, 100, 250, 30)
        self.input3.setStyleSheet('''
                                background:#DEDEDE;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.btn3 = QtWidgets.QPushButton(self)  # Button
        self.btn3.move(580, 100)
        self.btn3.setText('Browse')
        self.btn3.setStyleSheet('''
                                background:#77F2A1;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.btn3.clicked.connect(lambda: self.openFile(self.input3))

        # dataset_type
        self.label4 = QtWidgets.QLabel(self)  # Label
        self.label4.setText('For training or test dataset:')
        self.label4.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label4.move(5, 140)
        button_group4 = QtWidgets.QButtonGroup(self)
        self.rb_a4 = QtWidgets.QRadioButton(self)    # 單選
        button_group4.addButton(self.rb_a4)
        self.rb_a4.setGeometry(300, 145, 100, 20)
        self.rb_a4.setText('train')
        self.rb_a4.setChecked(True)  # default, dataset_type=train
        self.rb_a4.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.rb_b4 = QtWidgets.QRadioButton(self)
        button_group4.addButton(self.rb_b4)
        self.rb_b4.setGeometry(450, 145, 100, 20)
        self.rb_b4.setText('test')
        self.rb_b4.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')

        # format
        self.label5 = QtWidgets.QLabel(self)  # Label
        self.label5.setText('Converted format:')
        self.label5.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label5.move(5, 180)
        self.box5 = QtWidgets.QComboBox(self)  # 下拉選單
        self.box5.addItems(['coco', 'yoloSeg', 'yoloBbox'])
        self.box5.setCurrentIndex(0)  # default, Stride=1
        self.box5.setGeometry(200, 180, 200, 30)
        self.box5.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')

        # patch_size
        self.label6 = QtWidgets.QLabel(self)  # Label
        self.label6.setText('Patch size:')
        self.label6.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label6.move(5, 220)
        self.input6 = QtWidgets.QLineEdit(self)  # 單行輸入框
        self.input6.setText('1024')  # default, patch_size=1024
        self.input6.setGeometry(150, 220, 80, 30)
        self.input6.setStyleSheet('''
                                background:#DEDEDE;
                                color:#1F2B37;
                                font-size:18px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.input6.textChanged.connect(self.enable_disable_controls)   # 控制 Stride 和 store_none
        self.label6_tip = QtWidgets.QLabel(self)
        self.label6_tip.setText('(Optional, the input must be divided by the length and width of the original image)')
        self.label6_tip.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:14px;
                                font-family:monospace;
                                ''')
        self.label6_tip.move(250, 230)

        # Stride (有patch_size才生效)
        self.label7 = QtWidgets.QLabel(self)  # Label
        self.label7.setText('Stride:')
        self.label7.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label7.move(5, 260)
        self.box7 = QtWidgets.QComboBox(self)  # 下拉選單
        self.box7.addItems(['1', '1/2'])
        self.box7.setCurrentIndex(0)  # default, Stride=1
        self.box7.setGeometry(100, 260, 100, 30)
        self.box7.setStyleSheet('''
            QComboBox {
                color: #DEDEDE;
                font-size: 20px;
                font-weight: bold;
                font-family: monospace;
            }
            QComboBox:disabled {
                color: #808080;
            }
        ''')
        self.box7.view().setStyleSheet('''  
            QAbstractItemView {
                color: #DEDEDE;
                background-color: #1F2B37;
            }
        ''')    # 解決QComboBox時，無法實現 color: #DEDEDE
        self.label7_tip = QtWidgets.QLabel(self)
        self.label7_tip.setText('(Patch size is required)')
        self.label7_tip.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:14px;
                                font-family:monospace;
                                ''')
        self.label7_tip.move(220, 270)

        # store_none (有patch_size才生效)
        self.label8 = QtWidgets.QLabel(self)  # Label
        self.label8.setText('Whether to save images without defects:')
        self.label8.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:20px;
                                font-weight:bold;
                                font-family:monospace;
                                ''')
        self.label8.move(5, 300)
        button_group8 = QtWidgets.QButtonGroup(self)
        self.rb_a8 = QtWidgets.QRadioButton(self)  # 單選
        button_group8.addButton(self.rb_a8)
        self.rb_a8.setGeometry(450, 305, 100, 20)
        self.rb_a8.setText('Yes')
        self.rb_a8.setStyleSheet('''
                                QRadioButton {
                                    color:#DEDEDE;
                                    font-size:20px;
                                    font-weight:bold;
                                    font-family:monospace;
                                }
                                QRadioButton:disabled {
                                    color:#808080;
                                }
                                ''')
        self.rb_b8 = QtWidgets.QRadioButton(self)
        button_group8.addButton(self.rb_b8)
        self.rb_b8.setGeometry(550, 305, 100, 20)
        self.rb_b8.setText('No')
        self.rb_b8.setChecked(True)  # default, store_none=No
        self.rb_b8.setStyleSheet('''
                                QRadioButton {
                                    color:#DEDEDE;
                                    font-size:20px;
                                    font-weight:bold;
                                    font-family:monospace;
                                }
                                QRadioButton:disabled {
                                    color:#808080;
                                }
                                ''')
        self.label8_tip = QtWidgets.QLabel(self)
        self.label8_tip.setText('(Patch size is required)')
        self.label8_tip.setStyleSheet('''
                                color:#DEDEDE;
                                font-size:14px;
                                font-family:monospace;
                                ''')
        self.label8_tip.move(620, 305)

        # Start -> create data
        self.run_btn = QtWidgets.QPushButton(self)  # Button
        self.run_btn.setGeometry(340, 370, 80, 30)
        self.run_btn.setText('Start')
        self.run_btn.setStyleSheet('''
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
        self.run_btn.clicked.connect(self.run_program)

    def openFolder(self, input_line):
        folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        input_line.setText(folderPath)
        # print(folderPath)

    def openFile(self, input_line):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName()
        input_line.setText(filePath)
        # print(filePath)

    def enable_disable_controls(self):
        patch_size_specified = bool(self.input6.text())
        self.box7.setDisabled(not patch_size_specified)
        self.rb_a8.setDisabled(not patch_size_specified)
        self.rb_b8.setDisabled(not patch_size_specified)

    def run_program(self):
        source_dir = self.input1.text()
        output_dir = self.input2.text()
        classes_yaml = self.input3.text()
        dataset_type = "train" if self.rb_a4.isChecked() else "test"
        output_format = self.box5.currentText()
        patch_size = self.input6.text() if self.input6.text() else None
        stride = 2 if self.box7.currentText() == '1/2' else int(self.box7.currentText())
        store_none = self.rb_a8.isChecked()

        # Error message
            # Check path
        if not all(os.path.exists(path) for path in [source_dir, output_dir, classes_yaml]):
            error_path = "One or more specified paths do not exist."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_path}</font>')
            return
            # Check patch size
        elif patch_size and (patch_size.isdigit() == False or int(patch_size) <= 0):
            error_int = "Patch size must be a positive integer."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_int}</font>')
            return
        elif patch_size and any(size % int(patch_size) != 0 for size in self.get_image_sizes(source_dir)):
            error_divisible = "Patch size must be divisible by the width and height of the original image."
            QtWidgets.QMessageBox.warning(self, "Error", f'<font size=4 color=#DEDEDE>{error_divisible}</font>')
            return


        cmd = [
            "python",
            "./tools/create_data.py",    ##### 需要更改!!!!!
            "--source_dir", source_dir,
            "--output_dir", output_dir,
            "--classes_yaml", classes_yaml,
            "--dataset_type", dataset_type,
            "--format", output_format,
        ]
        if patch_size is not None:
            cmd.extend(['--patch_size', str(patch_size)])  # Maybe None
            if stride is not None:
                cmd.extend(['--stride', str(stride)])  # Maybe None
            if store_none is not None:
                cmd.append("--store_none")

        # Run
        subprocess.run(cmd, check=True)


    def get_image_sizes(selg, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(folder_path, filename)
                with Image.open(image_path) as img:
                    width, height = img.size
                    return width, height
        return None


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec())