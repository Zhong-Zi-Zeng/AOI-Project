from __future__ import annotations
import cv2
import json
import os
import numpy as np


class JsonParser:
    """
        將傳入的json檔路徑進行解析，可取得json檔中想要的資訊
    """
    def __init__(self, json_file_path: str):
        assert os.path.isfile(json_file_path)

        self.json_file_path = json_file_path

        with open(json_file_path, 'r') as file:
            self.text = json.loads(file.read())

    def check_json(self) -> bool:
        """檢查json檔中是否存在想要的資訊"""
        if self.parse() is None:
            return False
        return True

    def parse(self) -> [tuple | None]:
        """
            解析出json檔中的mask、cls、bboxes、polygons等資訊,
            若其中一個資訊有問題, 則返回None

            Returns:
                如果有任何錯誤則返回None, N為瑕疵的數量, M為polygon的數量

                h (int): 圖片的高
                w (int): 圖片的寬
                mask (np.ndarray):  0、255的二值化影像 [H, W]
                classes (list): 這個json檔中包含的瑕疵類別 [N, ]
                bboxes (list): 每個瑕疵對應的bbox，格式為x, y, w, h [N, 4]
                polygons (list[np.ndarray]): 每個瑕疵對應的polygon [N, M, 2]
        """
        try:
            objects = self.text['Objects']
            image_height = int(self.text['Background']['Height'])  # 照片的高
            image_width = int(self.text['Background']['Width'])  # 照片的寬

            mask = np.zeros(shape=(image_height, image_width))
            classes = []
            bboxes = []
            polygons = []

            for obj in objects:
                # Mask
                polygon = obj['Layers'][0]['Shape']['Points']
                polygon = np.array([list(map(float, poly.split(','))) for poly in polygon], dtype=np.int32)
                cv2.fillPoly(mask, [polygon], (255, 255, 255))

                # polygons
                polygons.append(polygon)

                # classes
                classes.append(obj['Class']['$ref'])

                # bboxes
                x, y, w, h = cv2.boundingRect(polygon)
                bboxes.append([x, y, w, h])

        except KeyError as key:
            print('{} can\'t find the key of {}'.format(self.json_file_path, key))
            return None
        except Exception as e:
            print(e)
            return None

        return image_height, image_width, mask, classes, bboxes, polygons
