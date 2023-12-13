from __future__ import annotations
from abc import ABC, abstractmethod
import os


class BaseModel(ABC):
    def __init__(self, result_dir: str, output_dir: str,):

        # Generate a folder to store the results of training curve
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def generate_curve(self):
        pass
