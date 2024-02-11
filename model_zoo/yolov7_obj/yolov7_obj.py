from __future__ import annotations, absolute_import
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'yolov7_obj'))
from typing import Union, Any
from models.experimental import attempt_load
from utils.plots import plot_one_box
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from model_zoo.base.BaseDetectModel import BaseDetectModel
from engine.timer import TIMER
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, check_path)
from utils.torch_utils import select_device
import numpy as np
import cv2
import torch
import random
import subprocess


class Yolov7Obj(BaseDetectModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg

    def _config_transform(self):
        # Update data file
        data_file = load_yaml(self.cfg['data_file'])
        data_file['train'] = os.path.join(os.getcwd(), self.cfg['train_txt'])
        data_file['val'] = os.path.join(os.getcwd(), self.cfg['val_txt'])
        data_file['nc'] = self.cfg['number_of_class']
        data_file['names'] = self.cfg['class_names']
        self.cfg['data_file'] = os.path.join(get_work_dir_path(self.cfg), 'data.yaml')
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'data.yaml'), data_file)

        # Update hyp file
        hyp_file = load_yaml(self.cfg['hyp_file'])
        hyp_file['lr0'] = self.cfg['initial_lr'] / self.cfg['warmup_epoch'] / hyp_file['warmup_bias_lr']
        hyp_file['lrf'] = self.cfg['minimum_lr'] / hyp_file['lr0']
        hyp_file['hsv_h'] = self.cfg['hsv_h']
        hyp_file['hsv_s'] = self.cfg['hsv_s']
        hyp_file['hsv_v'] = self.cfg['hsv_v']
        hyp_file['degrees'] = self.cfg['degrees']
        hyp_file['translate'] = self.cfg['translate']
        hyp_file['scale'] = self.cfg['scale']
        hyp_file['shear'] = self.cfg['shear']
        hyp_file['perspective'] = self.cfg['perspective']
        hyp_file['flipud'] = self.cfg['flipud']
        hyp_file['fliplr'] = self.cfg['fliplr']
        hyp_file['mosaic'] = self.cfg['mosaic']
        hyp_file['mixup'] = self.cfg['mixup']
        hyp_file['copy_paste'] = self.cfg['copy_paste']
        self.cfg['hyp_file'] = os.path.join(get_work_dir_path(self.cfg), 'hyp.yaml')
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'hyp.yaml'), hyp_file)

        # Update cfg file
        cfg_file = load_yaml(self.cfg['cfg_file'])
        cfg_file['nc'] = self.cfg['number_of_class']
        self.cfg['cfg_file'] = os.path.join(get_work_dir_path(self.cfg), 'cfg.yaml')
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'cfg.yaml'), cfg_file)

    def _load_model(self):
        # Load model
        self.device = select_device(self.cfg['device'])
        self.model = attempt_load(self.cfg['weight'], map_location=self.device)
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.cfg['imgsz'][0], s=self.stride)  # check img_size

        # Warmup
        for i in range(3):
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    def train(self):
        """
            Run每個model自己的training command
        """

        subprocess.run(['python',
                        os.path.join(get_model_path(self.cfg), 'train.py'),
                        '--data', self.cfg['data_file'],
                        '--cfg', self.cfg['cfg_file'],
                        '--hyp', self.cfg['hyp_file'],
                        '--batch-size', str(self.cfg['batch_size']),
                        '--weights', self.cfg['weight'] if check_path(self.cfg['weight']) else " ",
                        '--epochs', str(self.cfg['end_epoch'] - self.cfg['start_epoch']),
                        '--project', get_work_dir_path(self.cfg),
                        '--optimizer', self.cfg['optimizer'],
                        '--device', self.cfg['device'],
                        '--name', './',
                        '--save_period', str(self.cfg['save_period']),
                        '--eval_period', str(self.cfg['eval_period']),
                        '--img-size', str(self.cfg['imgsz'][0]),
                        '--exist-ok',
                        ]
                       )

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
                # Load image
                if isinstance(source, str):
                    original_image = cv2.imread(source)
                elif isinstance(source, np.ndarray):
                    original_image = source
                else:
                    raise ValueError

                # Transform image
                im = original_image.copy()
                im = letterbox(im, self.imgsz, stride=self.stride)[0]
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous
                im = torch.from_numpy(im).to(self.device)
                im = im.float()
                im /= 255.  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # ----------------------------Pre-process (End)----------------------------

            # ----------------------------Inference (Start))----------------------------
            # Inference
            with TIMER[2], torch.no_grad():
                pred = self.model(im)[0]
            # ----------------------------Inference (End)----------------------------

            # ----------------------------NMS-process (Start)----------------------------
            with TIMER[3]:
                pred = non_max_suppression(pred, conf_thres, nms_thres, classes=None, agnostic=False)
            # ----------------------------NMS-process (End)----------------------------

            # ----------------------------Post-process (Start)----------------------------
            class_list = []
            score_list = []
            bbox_list = []

            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], original_image.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        cls = int(cls.cpu())
                        conf = float(conf.cpu())

                        x = xyxy[0].cpu().numpy()
                        y = xyxy[1].cpu().numpy()
                        w = xyxy[2].cpu().numpy() - x
                        h = xyxy[3].cpu().numpy() - y

                        bbox_list.append(list(map(float, [x, y, w, h])))
                        class_list.append(cls)
                        score_list.append(conf)

                        # Draw bounding box
                        text = f'{self.class_names[int(cls)]} {conf:.2f}'
                        self.plot_one_box_mask(image=original_image,
                                               xywh_bbox=[x, y, w, h],
                                               text=text,
                                               color=self.class_color[int(cls)])

            return {
                'result_image': original_image,
                'class_list': class_list,
                'score_list': score_list,
                'bbox_list': bbox_list
            }
