from __future__ import annotations

import subprocess
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


custom_config_path = "./src/anomalib/models/efficient_ad/config.yaml"
modified_config_path = "./src/anomalib/models/efficient_ad/config_modified.yaml"

abnormal_dir_class_4 = ["test/defect/Assembly", "test/defect/Friction", "test/defect/Dirty", "test/defect/Scratch"]
mask_dir_class_4 = ["mask/defect/Assembly", "mask/defect/Friction", "mask/defect/Dirty", "mask/defect/Scratch"]
abnormal_dir_class_1 = ["test/defect/Defect"]
mask_dir_class_1 = ["mask/defect/Defect"]
batch_size = 4
image_size = 256
project_path = "./results/white_controller_selected"
max_epochs = 50

experiment_order = [
    # {"name": "original_class_4",
    #  "path": "./datasets/white_controller_selected/original_class_4",
    #  "abnormal_dir": abnormal_dir_class_4,
    #  "mask_dir": mask_dir_class_4},

    # {"name": "original_class_1",
    #  "path": "./datasets/white_controller_selected/original_class_1",
    #  "abnormal_dir": abnormal_dir_class_1,
    #  "mask_dir": mask_dir_class_1},

    {"name": "patch_1024_class_4",
     "path": "./datasets/white_controller_selected/patch_1024_class_4",
     "abnormal_dir": abnormal_dir_class_4,
     "mask_dir": mask_dir_class_4},

    {"name": "patch_1024_class_1",
     "path": "./datasets/white_controller_selected/patch_1024_class_1",
     "abnormal_dir": abnormal_dir_class_1,
     "mask_dir": mask_dir_class_1},

    {"name": "patch_512_class_4",
     "path": "./datasets/white_controller_selected/patch_512_class_4",
     "abnormal_dir": abnormal_dir_class_4,
     "mask_dir": mask_dir_class_4},

    {"name": "patch_512_class_1",
     "path": "./datasets/white_controller_selected/patch_512_class_1",
     "abnormal_dir": abnormal_dir_class_1,
     "mask_dir": mask_dir_class_1},

    {"name": "patch_256_class_4",
     "path": "./datasets/white_controller_selected/patch_256_class_4",
     "abnormal_dir": abnormal_dir_class_4,
     "mask_dir": mask_dir_class_4},

    {"name": "patch_256_class_1",
     "path": "./datasets/white_controller_selected/patch_256_class_1",
     "abnormal_dir": abnormal_dir_class_1,
     "mask_dir": mask_dir_class_1}
]



if __name__ == "__main__":
    for idx, experiment in enumerate(experiment_order):
        config = load_yaml(custom_config_path)

        config['dataset']['name'] = experiment['name']
        config['dataset']['path'] = experiment['path']
        config['dataset']['abnormal_dir'] = experiment['abnormal_dir']
        config['dataset']['mask_dir'] = experiment['mask_dir']
        config['dataset']['train_batch_size'] = batch_size
        config['dataset']['eval_batch_size'] = batch_size
        config['dataset']['image_size'] = image_size
        config['project']['path'] = project_path
        config['trainer']['max_epochs'] = max_epochs

        save_yaml(modified_config_path, config)

        subprocess.run(['python', "./tools/train.py", '--config', modified_config_path])
