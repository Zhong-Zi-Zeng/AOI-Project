from mmdet.apis import DetInferencer
import pycocotools.mask as ms
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import matplotlib.patches as patches
import numpy as np
import cv2
import pycocotools.mask as mask


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
results_dict = inferencer(r"D:\Heng_shared\AOI-Project\data\white_controller\coco\original_class_1\val2017\0.jpg",
                          show=False, print_result=False, return_vis=True, pred_score_thr=0.5)

predictions = results_dict['predictions'][0]
vis = results_dict['visualization'][0]

classes = predictions['labels']
scores = predictions['scores']
bboxes = predictions['bboxes']
rles = predictions['masks']
fig, ax = plt.subplots(1)
image = cv2.imread(r"D:\Heng_shared\AOI-Project\data\white_controller\coco\original_class_1\val2017\0.jpg")

ax.imshow(image)

for cls, conf, bbox, rle in zip(classes, scores, bboxes, rles):
    if conf < 0.5:
        continue

    maskedArr = mask.decode(rle)
    polygons = polygonFromMask(mask_util.decode(rle))

    for polygon in polygons:
        poly = np.reshape(np.array(polygon), (-1, 2))
        color = list(np.random.random(size=(3,)))

        x, y, w, h = cv2.boundingRect(poly)

        polygon = patches.Polygon(poly, closed=True, fill=True, edgecolor='r', facecolor=color, alpha=0.5)
        ax.add_patch(polygon)

        ax.add_patch(plt.Rectangle((x, y), w, h,
                                   fill=False, color=color, linewidth=3))

        ax.text(x, y, cls_name[cls],
                bbox=dict(facecolor='yellow', alpha=0.5))

    # rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r',
    #                          facecolor='none')
    # ax.add_patch(rect)
    # plt.text(bbox[0], bbox[1], cls_name[cls], fontsize=12, color='r', verticalalignment='top')
plt.show()
# for mask in predictions['masks']:
#     polygon = ms.decode(mask)
#     print(polygon)
