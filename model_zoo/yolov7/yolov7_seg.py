from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'yolov7', 'yolov7_seg'))

from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2, non_max_suppression, scale_segments, scale_coords)
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device
from ..base.BaseModel import BaseModel
from engine.general import (get_work_dir_path, load_yaml, save_yaml)
import numpy as np
import torch
import yaml


class Yolov7Seg(BaseModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg
        self._config_transform()

    def _config_transform(self):
        # Update data file
        data_file = load_yaml(self.cfg['data_file'])
        data_file['train'] = self.cfg['train_txt']
        data_file['val'] = self.cfg['val_txt']
        data_file['nc'] = self.cfg['number_of_class']
        data_file['names'] = self.cfg['class_names']
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'data.yaml'), data_file)

        # Update hyp file
        hyp_file = load_yaml(self.cfg['hyp_file'])
        hyp_file['lr0'] = self.cfg['lr'] * self.cfg['start_factor']
        hyp_file['lrf'] = self.cfg['lr']
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'hyp.yaml'), hyp_file)

    def _load_model(self):
        # Load model
        self.device = select_device('')
        self.model = DetectMultiBackend(self.cfg['weight'],
                                        device=self.device,
                                        dnn=False,
                                        data=os.path.join(get_work_dir_path(self.cfg), 'data.yaml'),
                                        fp16=False)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.cfg['imgsz'], s=stride)  # check image size

        # Run inference
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt else bs, 3, *self.cfg['imgsz']))  # warmup

    def _predict(self,
                 source: [str | np.ndarray[np.uint8]],
                 conf_thres: float = 0.25,
                 nms_thres: float = 0.5,
                 *args,
                 **kwargs) -> dict:

        if not hasattr(self, 'model'):
            self._load_model()

        max_det = kwargs.get('max_det', 1000)
        line_thickness = kwargs.get('line_thickness', 3)

        with self.dt[0]:
            # ------------------------------Pre-process (Start)----------------------------
            with self.dt[1]:
                # Load image
                if isinstance(source, str):
                    original_image = cv2.imread(source)
                elif isinstance(source, np.ndarray):
                    original_image = source
                else:
                    raise ValueError

                # Transform image
                im = letterbox(original_image, self.imgsz)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous

                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # ----------------------------Pre-process (End)----------------------------

            # ----------------------------Inference (Start))----------------------------
            # Inference
            with self.dt[2]:
                pred, out = self.model(im)
                proto = out[1]
            # ----------------------------Inference (End)----------------------------

            # ----------------------------NMS-process (Start)----------------------------
            with self.dt[3]:
                pred = non_max_suppression(pred, conf_thres, nms_thres, classes=None, agnostic=False, max_det=max_det,
                                           nm=32)
            # ----------------------------NMS-process (End)----------------------------

            # ----------------------------Post-process (Start)----------------------------
            # For evaluation
            class_list = []
            score_list = []
            bbox_list = []
            polygon_list = []

            # Process predictions
            annotator = Annotator(original_image, line_width=line_thickness, example=str(self.names))

            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Process mask
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], original_image.shape).round()

                    # Segments
                    segments = reversed(masks2segments(masks))
                    segments = [scale_segments(im.shape[2:], x, original_image.shape).round() for x in segments]

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                    im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                    annotator.im = scale_masks(im.shape[2:], im_masks, original_image.shape)  # scale to original h, w
                    # Mask plotting ----------------------------------------------------------------------------------------

                    # Record result
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        cls = int(cls.cpu())
                        conf = float(conf.cpu())

                        x = xyxy[0].cpu().numpy()
                        y = xyxy[1].cpu().numpy()
                        w = xyxy[2].cpu().numpy() - x
                        h = xyxy[3].cpu().numpy() - y

                        bbox_list.append(list(map(float, [x, y, w, h])))
                        class_list.append(cls)
                        score_list.append(conf)
                        polygon_list.append(segments[j])

                        # Draw bounding box
                        annotator.box_label(xyxy, self.names[cls], color=colors(cls, True))

            # results
            result_image = annotator.result()

            # ----------------------------Post-process (End)----------------------------

        return {"result_image": result_image,
                "class_list": class_list,
                "bbox_list": bbox_list,
                "score_list": score_list,
                "polygon_list": polygon_list}

    def train(self):
        pass
