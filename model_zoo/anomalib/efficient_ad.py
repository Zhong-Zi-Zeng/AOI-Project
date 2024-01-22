from __future__ import annotations, absolute_import

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'anomalib', 'src'))
from typing import Union, Any
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from model_zoo.base.BaseDetectModel import BaseDetectModel
from engine.timer import TIMER
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path)

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

import numpy as np
import cv2
import subprocess


class EfficientAD(BaseDetectModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg

    def _config_transform(self):
        # 讀取程式中的config
        config = load_yaml(self.cfg['cfg_file'])

        # 更新custom調整的config
        config['dataset']['path'] = self.cfg['dataset_dir']

        config['dataset']['abnormal_dir'] = []  # Clear
        config['dataset']['mask_dir'] = []
        for idx, cls_name in enumerate(self.cfg['class_names']):
            config['dataset']['abnormal_dir'].append(f'test/defect/{cls_name}')
            config['dataset']['mask_dir'].append(f'mask/defect/{cls_name}')

        config['model']['lr'] = self.cfg['lr']
        config['project']['path'] = os.path.join(get_work_dir_path(self.cfg))  # result
        config['trainer']['min_epochs'] = self.cfg['start_epoch']
        config['trainer']['max_epochs'] = self.cfg['end_epoch']
        config['dataset']['train_batch_size'] = self.cfg['batch_size']
        config['dataset']['eval_batch_size'] = self.cfg['batch_size']
        config['dataset']['image_size'] = self.cfg['imgsz'][0]
        config['trainer']['num_nodes'] = self.cfg['device']
        # config['trainer']['optimizer'] = self.cfg['optimizer']

        # 修改後的config儲存在work_dir
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'cfg.yaml'), config)
        self.cfg['cfg_file'] = os.path.join(get_work_dir_path(self.cfg), 'cfg.yaml')

    def train(self):
        """
            Run每個model自己的training command
        """
        subprocess.run([
            'python', os.path.join(get_model_path(self.cfg), 'tools', 'train.py'),
            '--config', self.cfg['cfg_file']
        ])

    def _load_model(self):
        """
            載入model，為後面的inference或是evaluate使用
        """
        config = get_configurable_parameters(config_path=self.cfg['cfg_file'])
        config.trainer.resume_from_checkpoint = str(self.cfg['weight'])
        config.visualization.show_images = False
        config.visualization.mode = "simple"

        # create model
        self.model = get_model(config)
        self.model.to(int(self.cfg['device']))
        self.model.eval()

        # create trainer
        callbacks = get_callbacks(config)
        self.trainer = Trainer(callbacks=callbacks, **config.trainer)

        # get the transforms
        transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
        image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
        center_crop = config.dataset.get("center_crop")
        if center_crop is not None:
            center_crop = tuple(center_crop)
        normalization = InputNormalizationMethod(config.dataset.normalization)
        self.transform = get_transforms(
            config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
        )

        self.config = config

    def _predict(self,
                 source: Union[str | np.ndarray[np.uint8]],
                 conf_thres: float = 0.25,
                 nms_thres: float = 0.5,
                 *args: Any,
                 **kwargs: Any
                 ) -> dict:

        if not hasattr(self, 'model'):
            self._check_weight_path(self.cfg['weight'])
            self._load_model()

        with TIMER[0]:
            # ------------------------------Pre-process (Start)----------------------------
            with TIMER[1]:
                print(source)
                # Load image
                # create the dataset
                dataset = InferenceDataset(
                    source, image_size=tuple(self.config.dataset.image_size), transform=self.transform  # type: ignore
                )
                dataloader = DataLoader(dataset)

                # Image preprocessing
            # ----------------------------Pre-process (End)----------------------------

            # ----------------------------Inference (Start))----------------------------
            # Inference
            with TIMER[2]:
                # generate predictions
                result = self.trainer.predict(model=self.model, dataloaders=[dataloader])[0]    # 一次處理一張
                # print(result)   # image, image_path, anomaly_maps, pred_scores, pred_labels, pred_masks, pred_boxes

            # ----------------------------Inference (End)----------------------------

            # ----------------------------NMS-process (Start)----------------------------
            with TIMER[3]:
                pass
            # ----------------------------NMS-process (End)----------------------------

            # ----------------------------Post-process (Start)----------------------------
            # For evaluation
            class_list = []
            score_list = []
            bbox_list = []

            # original images
            origin = result['image'].squeeze().permute(1, 2, 0).detach().cpu().numpy()
            origin = np.array(origin * 255, dtype=np.uint8)
            origin = np.clip(origin, 0, 255)
            # cv2.imwrite('./work_dirs/test!!/origin.png', origin)

            # mask
            pred_masks = result['pred_masks']
            pred_masks_np = pred_masks.squeeze().detach().cpu().numpy()
            pred_masks_np = np.array(pred_masks_np * 255, dtype=np.uint8)
            pred_masks_np = np.clip(pred_masks_np, 0, 255)
            # cv2.imwrite('./work_dirs/test!!/pred_masks.png', pred_masks_np)
            origin = np.ascontiguousarray(origin)

            # polygons
            _, thresh = cv2.threshold(pred_masks_np, conf_thres, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # bounding boxes
            scores = result['pred_scores']
            scores_float = [float(score) for score in scores]
            for score, contour in zip(scores_float, contours):
                if score > conf_thres:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(origin, (x, y), (x + w, y + h), (255, 255, 255), 2)

                    class_list.append(0)  # 只有一種
                    score_list.append(score)
                    bbox_list.append([x, y, w, h])

            # Save the image with bounding boxes
            # img_name = os.path.basename(res['image_path'][0])
            # output_path = os.path.join('./work_dirs/test!!/', f'bbox_{img_name}')
            # cv2.imwrite(output_path, origin)

            return {
                'result_image': origin,   # Labeled image
                'class_list': class_list,      # only one
                'score_list': score_list,     # confidence = 1
                'bbox_list': bbox_list
            }

'''
python ./tools/predict.py 
-c "./configs/efficient_ad_custom.yaml" 
-s "/home/miislab-server/YA/YA_share/AOI-Project/data/mvtec/original_class_4/test/defect/" 
# 到./defect 就能預測整個 Assembly
'''