from .yolov7_obj import Yolov7Obj


class Yolov7XObj(Yolov7Obj):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg
