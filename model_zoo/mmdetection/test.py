from mmdet.apis import DetInferencer
import pycocotools.mask as ms
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import matplotlib.patches as patches
import numpy as np
import cv2
import pycocotools.mask as mask
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def polygonFromMask(maskedArr):
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())

    return segmentation  # , [x, y, w, h], area


cls_name = ["Scratch", "Friction", "Dirty", "Assembly"]

# Initialize the DetInferencer
inferencer = DetInferencer(model=r"D:\Heng_shared\AOI-Project\work_dirs\train\CascadeMaskRCNN\cfg.py",
                           weights=r"D:\Heng_shared\AOI-Project\work_dirs\train\CascadeMaskRCNN\epoch_400.pth")

# Perform inference
results_dict = inferencer(r"D:\Heng_shared\AOI-Project\data\white_controller\coco\original_class_1\val\0.jpg",
                          show=False, print_result=False, return_vis=True, pred_score_thr=0.5)

predictions = results_dict['predictions'][0]
vis = results_dict['visualization'][0]

classes = predictions['labels']
scores = predictions['scores']
bboxes = predictions['bboxes']
rles = predictions['masks']
fig, ax = plt.subplots(1)
image = cv2.imread(r"D:\Heng_shared\AOI-Project\data\white_controller\coco\original_class_1\val\0.jpg")
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax.imshow(image)

for cls, conf, bbox, rle in zip(classes, scores, bboxes, rles):
    if conf < 0.5:
        continue

    maskedArr = mask.decode(rle)
    polygons = polygonFromMask(mask_util.decode(rle))

    for polygon in polygons:
        poly = np.reshape(np.array(polygon), (-1, 2))
        color = list(np.random.uniform(0, 255, size=(3,)))

        x, y, w, h = cv2.boundingRect(poly)

        cv2.fillPoly(image, [poly], color=color)

        cv2.putText(image, cls_name[cls], (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 1, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)

        # polygon = patches.Polygon(poly, closed=True, fill=True, edgecolor='r', facecolor=color, alpha=0.5)
        # ax.add_patch(polygon)
        #
        # ax.add_patch(plt.Rectangle((x, y), w, h,
        #                            fill=False, color=color, linewidth=3))
        #
        # ax.text(x, y, cls_name[cls],
        #         bbox=dict(facecolor='yellow', alpha=0.5))

# canvas = FigureCanvas(fig)  # 使用FigureCanvasAgg
# canvas.draw()
# width, height = fig.get_size_inches() * fig.get_dpi()
# image_matplotlib = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
# canvas = FigureCanvas(fig)  # 使用FigureCanvasAgg
# canvas.draw()
#
# # 從Matplotlib圖像中獲取NumPy數組（使用frombuffer）
# width, height = fig.get_size_inches() * fig.get_dpi()
# image_matplotlib = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

cv2.imshow('', image)
cv2.waitKey(0)
# plt.show()
# for mask in predictions['masks']:
#     polygon = ms.decode(mask)
#     print(polygon)
