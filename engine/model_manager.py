from engine.builder import Builder


class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
        return cls._instance

    def initialize_model(self, final_config):
        builder = Builder(yaml_dict=final_config, task='predict')
        cfg = builder.build_config()
        self.model = builder.build_model(cfg)

    def get_model(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Please call initialize_model first.")
        return self.model
