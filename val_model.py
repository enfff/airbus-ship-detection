import torch

from utils import *

augmentation_type = 'noaug'
id = 0

model_name, model_root, model_filepath, log_filepath = generate_paths(augmentation_type, id)

if __name__ == '__main__':
    ground_truths = np.load('rcnn_targets.npy',allow_pickle='TRUE')
    
    print(len(ground_truths))
    # print(type(ground_truths))
    # print(type(ground_truths[0]))
    # print(ground_truths[0])
    # print(ground_truths[0].keys())

    # for gt in ground_truths:
    #     print(f"{gt = }\n")