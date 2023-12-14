from __future__ import annotations
from abc import ABC, abstractmethod
import os


class BaseModel(ABC):
    def __init__(self, result_path: str, output_path: str,):

        # Generate a folder to store the results of training curve
        os.makedirs(output_path, exist_ok=True)

    @abstractmethod
    def generate_curve(self):
        pass
