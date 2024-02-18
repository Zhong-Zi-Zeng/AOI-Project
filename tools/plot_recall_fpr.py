import os
import sys

sys.path.append(os.getcwd())
from engine.general import load_yaml, save_yaml
from engine.builder import Builder
from evaluation import Evaluator
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

weight_dir = r"D:\Heng_shared\AOI-Project\work_dirs\train\YOLOv7_size_1824_w_aug\weights"
config_file = r"D:\Heng_shared\AOI-Project\work_dirs\train\YOLOv7_size_1824_w_aug\final_config.yaml"

x = list(range(0, 900, 25))

image_recall = []
image_fpr = []
defect_recall = []
defect_fpr = []
for weight in os.listdir(weight_dir):
    print('='*40 + weight + '=' * 40)
    weight_file = os.path.join(weight_dir, weight)

    builder = Builder(config_path=config_file, task='eval')
    config = builder.build_config()
    config['weight'] = weight_file

    model = builder.build_model(config)
    evaluator = Evaluator(model=model, cfg=config)
    result = evaluator.eval()

    image_recall.append(result[0])
    image_fpr.append(result[1])
    defect_recall.append(result[2])
    defect_fpr.append(result[3])

# 使用Seaborn绘制折线图
sns.set(style="darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

sns.lineplot(x=x, y=image_recall, ax=axes[0, 0])
axes[0, 0].set_title('Recall(image)')
axes[0, 0].set_xlabel('Epoch')

sns.lineplot(x=x, y=image_fpr, ax=axes[0, 1])
axes[0, 1].set_title('FPR(image)')
axes[0, 1].set_xlabel('Epoch')

sns.lineplot(x=x, y=defect_recall, ax=axes[1, 0])
axes[1, 0].set_title('Recall(Defect)')
axes[1, 0].set_xlabel('Epoch')

sns.lineplot(x=x, y=defect_fpr, ax=axes[1, 1])
axes[1, 1].set_title('FPR(Defect)')
axes[1, 1].set_xlabel('Epoch')

plt.tight_layout()
plt.savefig("result.png")
