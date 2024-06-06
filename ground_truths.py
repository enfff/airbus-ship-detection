import numpy as np

"""
    This script is used to read the numpy file created by faster_rcnn_dataset_maker.py
"""

ground_truths = np.load('rcnn_targets.npy',allow_pickle='TRUE')
print(f"{ground_truths[1] = }\n")
print(f"{ground_truths[2] = }\n")
print(f"{ground_truths[3] = }\n")
print(f"{ground_truths[4] = }\n")
print(f"{ground_truths[5] = }\n")
print(f"{ground_truths[6] = }\n")
print(f"{ground_truths[7] = }\n")
print(f"{ground_truths[8] = }\n")
print(f"{ground_truths[9] = }\n")

"""
ground_truths[1] = {'boxes': tensor([]), 'labels': tensor([])}

ground_truths[2] = {'boxes': tensor([]), 'labels': tensor([])}

ground_truths[3] = {'boxes': tensor([[0.4466, 0.6055, 0.5820, 0.6523]]), 'labels': tensor([1.])}

ground_truths[4] = {'boxes': tensor([[0.6094, 0.3737, 0.6393, 0.3997],
        [0.0859, 0.4909, 0.1094, 0.5026],
        [0.3346, 0.2266, 0.3971, 0.2409],
        [0.0924, 0.5026, 0.1016, 0.5039],
        [0.4297, 0.2318, 0.4805, 0.2565]]), 'labels': tensor([1., 1., 1., 1., 1.])}

ground_truths[5] = {'boxes': tensor([]), 'labels': tensor([])}

ground_truths[6] = {'boxes': tensor([[0.1810, 0.9635, 0.2305, 1.0000],
        [0.1693, 0.9531, 0.2174, 0.9922],
        [0.1237, 0.9245, 0.1445, 0.9336],
        [0.1237, 0.9310, 0.1380, 0.9375],
        [0.2526, 0.8607, 0.2604, 0.8659],
        [0.2344, 0.8190, 0.2422, 0.8307],
        [0.1263, 0.9180, 0.1445, 0.9271],
        [0.1445, 0.9219, 0.1562, 0.9284],
        [0.1589, 0.9831, 0.1758, 0.9987]]), 'labels': tensor([1., 1., 1., 1., 1., 1., 1., 1., 1.])}

ground_truths[7] = {'boxes': tensor([[0.4323, 0.0469, 0.4557, 0.0807],
        [0.4206, 0.0534, 0.4479, 0.0872]]), 'labels': tensor([1., 1.])}

ground_truths[8] = {'boxes': tensor([]), 'labels': tensor([])}

ground_truths[9] = {'boxes': tensor([]), 'labels': tensor([])}
"""