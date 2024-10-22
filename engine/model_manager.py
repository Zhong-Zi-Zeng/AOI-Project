import torch

from engine.builder import Builder
from engine.general import get_gpu_count


class ModelManager:
    def __init__(self):
        self.model_dict = {f"GPU:{device_id}": None for device_id in range(get_gpu_count())}

    def initialize_model(self,
                         config: dict,
                         task: str,
                         work_dir_name: str = None) -> dict:
        assert task in ['train', 'predict', 'eval']

        builder = Builder(yaml_dict=config, task=task, work_dir_name=work_dir_name)
        final_config = builder.build_config()
        device_id = int(final_config['device'])
        model = builder.build_model(final_config)

        self.model_dict[f"GPU:{device_id}"] = model

        return final_config

    def get_model(self, device_id: int):
        if self.model_dict[f"GPU:{device_id}"] is None:
            raise ValueError("Model is not initialized. Please call initialize_model first.")

        model = self.model_dict[f"GPU:{device_id}"]

        return model

    def release_model(self, device_id: int) -> None:
        self.model_dict[f"GPU:{device_id}"] = None
        torch.cuda.empty_cache()
