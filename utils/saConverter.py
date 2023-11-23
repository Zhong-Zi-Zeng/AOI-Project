from __future__ import annotations
from baseConverter import BaseConverter
from patchify import patchify
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from jsonParser import JsonParser
from typing import Optional
import cv2
import json
import os
import numpy as np
import argparse
import shutil


class saConverter(BaseConverter):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 patch_size: Optional[int] = None):
        super().__init__(source_dir, output_dir)
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self._generate_dir()

    def _generate_dir(self):
        os.mkdir(os.path.join(self.output_dir, 'train'))
        os.mkdir(os.path.join(self.output_dir, 'test'))
        os.mkdir(os.path.join(self.output_dir, 'train', 'image'))
        os.mkdir(os.path.join(self.output_dir, 'train', 'label'))
        os.mkdir(os.path.join(self.output_dir, 'test', 'image'))
        os.mkdir(os.path.join(self.output_dir, 'test', 'label'))

    def generate_original(self):
        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            h, w, mask, classes, bboxes, polygons = JsonParser(json_file).parse()

            # image
            shutil.copy(image_file,
                        os.path.join(self.output_dir, 'train', 'image', str(idx) + '.jpg'))

            # label
            cv2.imwrite(os.path.join(self.output_dir, 'train', 'label', str(idx) + '.jpg'),
                        mask)

    def generate_patch(self):
        pass


a = saConverter(source_dir='./source', output_dir='./black').generate_original()
