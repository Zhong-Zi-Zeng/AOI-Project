from engine.builder import Builder


class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
        return cls._instance

    def initialize_model(self,
                         config: dict,
                         task: str,
                         work_dir_name: str = None) -> dict:

        assert task in ['train', 'predict']

        builder = Builder(yaml_dict=config, task=task, work_dir_name=work_dir_name)
        final_config = builder.build_config()
        self.model = builder.build_model(final_config)

        return final_config

    def get_model(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Please call initialize_model first.")
        return self.model
