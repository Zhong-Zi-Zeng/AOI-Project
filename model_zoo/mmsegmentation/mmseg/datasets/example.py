from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ExampleDataset(BaseSegDataset):
    METAINFO = dict(
        classes=["Scratch", "Friction", "Dirty", "Assembly"],
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192]])

    def __init__(self,
                 img_suffix='.jpg',
                 **kwargs):

        super().__init__(
            img_suffix=img_suffix,
            **kwargs)

