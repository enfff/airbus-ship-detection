import torch

from utils import *

augmentation_type = 'noaug'
id = 0

model_name, model_root, model_filepath, log_filepath = generate_paths(augmentation_type, id)

if __name__ == '__main__':
    plot_ground_truths(model_root=model_root)