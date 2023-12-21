from mmdet.apis import DetInferencer
import pycocotools.mask as ms

# Initialize the DetInferencer
inferencer = DetInferencer(model=r"C:\Users\鐘子恒\AppData\Roaming\JetBrains\PyCharmCE2023.1\scratches\123.yaml",
                           weights=r"\\DESKTOP-PPOB8AK\share\AOI_result\Instance Segmentation\cascade-mask-rcnn\1024_AdamW_202312071914\epoch_50.pth")

# Perform inference
results_dict = inferencer(r"D:\Heng_shared\coco\val2017\0.jpg", show=False, print_result=False, return_vis=True)

predictions = results_dict['predictions'][0]
vis = results_dict['visualization'][0]

labels = predictions['labels']
scores = predictions['scores']
bboxes = predictions['bboxes']
for mask in predictions['masks']:
    polygon = ms.decode(mask)
    print(polygon)
