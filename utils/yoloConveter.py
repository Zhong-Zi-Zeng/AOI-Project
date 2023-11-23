from __future__ import annotations
from baseConverter import BaseConverter
from patchify import patchify
from pathlib import Path
from jsonParser import JsonParser
from typing import Optional
import cv2
import json
import os
import numpy as np
import shutil


class yoloConverter(BaseConverter):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 patch_size: Optional[int] = None):

        super().__init__(source_dir, output_dir)
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.patch_size = patch_size

    def generate_original(self):
        pass

    def generate_patch(self):
        pass

a = yoloConverter(source_dir='./source', output_dir='./black')
