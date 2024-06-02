import torch
import os
import matplotlib.pyplot as plt

import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.io import read_image
from torchvision import transforms

from Plotter import Plotter

def new_model():
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

model = new_model()

checkpoint=torch.load('model.tar',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

# Plot Images
plotter = Plotter(model)
with torch.no_grad():
    for file in os.listdir('imgs/src/'):
        # Read Image
        image = read_image('imgs/src/'+file).float()/256.0
        # image = transform(image)
        plotter(image)
    