import sys
import os

sys.path.append(os.path.join(os.getcwd()))

from moodle_zoo import Yolov7

yolov7 = Yolov7(
    weights=r"\\DESKTOP-PPOB8AK\share\AOI_result\Instance Segmentation\yolov7\1024_SGD_202312061402\weights\best.pt",
    data=r"D:\Heng_shared\yolov7-segmentation\data\custom.yaml",
    imgsz=(1024, 1024)
)

yolov7.run(source=r"C:\Users\鐘子恒\Desktop\Side-Project\AOI_Project\tools\coco\train2017\2.jpg")

print(yolov7.timer())
