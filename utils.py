import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

def generate_paths(augmentation_type: str, id: str = '0') -> tuple[str, str, str, str]:
    model_name = 'model_'+ augmentation_type + '_id' + str(id)
    model_root = os.path.join('models', model_name)
    model_filepath = os.path.join(model_root,'model.tar')
    log_filepath = os.path.join(model_root,'log.txt')

    return model_name, model_root, model_filepath, log_filepath

def new_model():
    """
        Generates a faster RCNN model with a custom head, and additional batch normalization layers
    """

    model_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    for module in model_rcnn.backbone.body.modules():
        if isinstance(module, nn.Conv2d):
            # Insert batch normalization after convolutional layers
            module = nn.Sequential(
                module,
                nn.BatchNorm2d(module.out_channels),
                nn.ReLU(inplace=True)
            )

    for name, param in model_rcnn.named_parameters():
          param.requires_grad = False

    num_classes = 2 # background, ship
    in_features = model_rcnn.roi_heads.box_predictor.cls_score.in_features
    model_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model_rcnn


def elapsed_time(log_filepath: str):
    import re
    from datetime import datetime

    """
        Prints the elapsed time for the training model
            log_filepath: str, path to the log file
    """

    lines = []
    with open(log_filepath, 'r') as file:
        lines_buf = file.readlines()
        pattern = re.compile(r"E: \d+ B: \d+")
        pattern = re.compile(r"Epoch: \d+, Training Loss: \d+\.\d+, Validation Loss: \d+\.\d+, lr: \d+\.\d+")
        lines = [line for line in lines_buf if pattern.search(line)]

    def calculate_elapsed_time(lines=lines):
        start_time = datetime.strptime(lines[0].split(' ')[0] + ' ' + lines[0].split(' ')[1], '%m-%d %H:%M:%S')
        end_time = datetime.strptime(lines[-1].split(' ')[0] + ' ' + lines[-1].split(' ')[1], '%m-%d %H:%M:%S')
        return end_time - start_time

    print(f"(TRAIN) Total elapsed time: {calculate_elapsed_time()} (hours)")
    print(f"(TRAIN) Mean batch training time: {float(calculate_elapsed_time().total_seconds()/len(lines)).__round__(4)} (secs)")

def plot_results(model_filepath: str):
    """
        Plots the training results of the model, and saves the plot to the media folder
            model_filepath: str, path to the model file (.tar)
    """

    import matplotlib.pyplot as plt
    
    checkpoint = torch.load(model_filepath, map_location=torch.device('cpu'))
    model_name = model_filepath.split('/')[-2]

    lrs = checkpoint['lrs']
    validation_losses = checkpoint['validation_losses']
    training_losses = checkpoint['training_losses']
    epochs = range(checkpoint['epoch'] + 1)

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle('Training Results')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_ylabel('Learning Rates (logarithmic scale)')
    ax1.set_xlabel('Epoch')
    ax1.plot(epochs, lrs)
    ax1.legend(['Learning Rates'])

    ax2.grid(True)
    ax2.set_ylabel('Losses')
    ax2.set_xlabel('Epoch')
    ax2.plot(epochs, training_losses, validation_losses)
    ax2.legend(['Training Losses', 'Validation Losses'])

    plt.show()

    media_filepath = os.path.join(os.getcwd(), "media", model_name)
    os.makedirs(media_filepath, exist_ok=True)
    fig.savefig(os.path.join(media_filepath, "results.png"))

class Plotter:
    def __init__(self, model):
        self.__model = model
        if self.__model.training:
            self.__model.eval()

    def __call__(self, img):
        prediction = self.__model([img])[0]
        boxes = prediction['boxes'].int()
        scores = prediction['scores'].tolist()

        print(f"{scores = }")
        print(f"{boxes = }")
        
        num = len(boxes)
        if num > 0:
            img = draw_bounding_boxes(
                (img*256).byte(),
                boxes, 
                labels=['{:.2f}'.format(score*100) for score in scores],
                width = 1,
                colors = 'yellow',
                font='/usr/share/fonts/cantarell/Cantarell-VF.otf', # ad enf non trova arial
                #font='arial',
                font_size = 15
            )
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        ax.imshow(img.byte().permute(1, 2, 0))
        plt.show()
        plt.close()

def print_ground_truths(indices: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
        Prints the ground truths computed in faster_rcnn_dataset_maker.py
        An entry of such object is as follows:
            ground_truths[4] = {'boxes': tensor([[0.6094, 0.3737, 0.6393, 0.3997],
            [0.0859, 0.4909, 0.1094, 0.5026],
            [0.3346, 0.2266, 0.3971, 0.2409],
            [0.0924, 0.5026, 0.1016, 0.5039],
            [0.4297, 0.2318, 0.4805, 0.2565]]), 'labels': tensor([1., 1., 1., 1., 1.])}
    """
    
    ground_truths = np.load('rcnn_targets.npy',allow_pickle='TRUE')
    # print(f"{ground_truths.shape = }")
    print(f"{len(ground_truths)}")

    for index in indices:
        print(f"{ground_truths[index] = }\n")