from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy
from jsonParser import JsonParser
import os


class BaseConverter(ABC):
    def __init__(self, source_dir: str, output_dir: str):
        # 生成初始的資料夾
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'delete'))

        self.image_files_path = None  # 儲存所有正確的image路徑
        self.json_files_path = None  # 儲存所有正確的image路徑

        self._check_file(source_dir, output_dir)

    @abstractmethod
    def generate_original(self):
        pass

    @abstractmethod
    def generate_patch(self):
        pass

    @staticmethod
    def is_image(image_path: str) -> bool:
        """
           判斷該路徑是不是圖像
           Arg:
                image_path: 影像路徑
           Return:
                True or False
       """
        assert os.path.isfile(image_path), 'Can\'t find the file {}.'.format(image_path)

        _allow_format = ['.jpg', '.png', '.bmp']
        return Path(image_path).suffix in _allow_format

    @staticmethod
    def is_json(json_path: str) -> bool:
        """
            判斷該路徑是不是json檔
            Arg:
                json_path: json檔路徑
            Return:
                True or False
        """
        assert os.path.isfile(json_path), 'Can\'t find the file {}.'.format(json_path)
        return Path(json_path).suffix == '.json'

    def _check_file(self, source_dir: str, output_dir: str):
        """
            對source資料夾下的圖片和json進行配對, 若有問題的檔案則會被移動到
            output_dir/delete 資料夾下
        """
        # 把source中所有的image和json取出
        self.image_files_path = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)
                                 if self.is_image(os.path.join(source_dir, image_name))]

        self.json_files_path = [os.path.join(source_dir, json_name) for json_name in os.listdir(source_dir)
                                if self.is_json(os.path.join(source_dir, json_name))]

        image_files_copy = deepcopy(self.image_files_path)
        json_files_copy = deepcopy(self.json_files_path)

        # 紀錄類別數量
        classes = {}

        # 對檔案進行匹配
        for image_file in self.image_files_path:
            # 尋找對應的json檔
            correspond_json_file = image_file + '.json'

            # 檢查對應的json檔是否有在source資料夾下
            if correspond_json_file not in self.json_files_path:
                continue

            # 檢查對應的json檔內容是否正確
            if not JsonParser(correspond_json_file).check_json():
                continue

            # 將對應到的檔案從copy list中移除
            image_files_copy.remove(image_file)
            json_files_copy.remove(correspond_json_file)

        # 把剩下在的檔案移動到delete資料夾下
        for except_image_file in image_files_copy:
            os.rename(except_image_file, os.path.join(output_dir, 'delete', Path(except_image_file).name))
        for except_json_file in json_files_copy:
            os.rename(except_json_file, os.path.join(output_dir, 'delete', Path(except_json_file).name))

        # 把最後正確的紀錄下來
        self.image_files_path = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)
                                 if self.is_image(os.path.join(source_dir, image_name))]

        self.json_files_path = [os.path.join(source_dir, json_name) for json_name in os.listdir(source_dir)
                                if self.is_json(os.path.join(source_dir, json_name))]
