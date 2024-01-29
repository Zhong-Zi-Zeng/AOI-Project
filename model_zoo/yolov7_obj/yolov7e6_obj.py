from .yolov7w6_obj import Yolov7W6Obj


class Yolov7E6Obj(Yolov7W6Obj):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg
