import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

def new_model():
    model_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

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


if __name__ == "__main__":
    # model_filepath = "/home/enf/Projects/airbus-ship-detection/models/model_epochs10_noaug_id0/model.tar"
    model_name = "model_noaug_id0"
    model_filepath = os.path.join("models", model_name, "model.tar")
    # path = os.path.join(model_filepath, "model.tar")
    checkpoint = torch.load(model_filepath, map_location=torch.device('cpu'))

    model = new_model()

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

    # ax3.grid(True)
    # ax3.set_title('Learning Rates (lrs)')
    # ax3.set_xlabel('Epoch')
    # ax3.plot(epochs, lrs)

    plt.show()

    media_filepath = os.path.join(os.getcwd(), "media", model_name)
    os.makedirs(media_filepath, exist_ok=True)
    plt.savefig(os.path.join(media_filepath, "results.png"))

    # ax2.grid(True)
    # ax2.set_title('Validation Losses')
    # ax2.set_xlabel('Epoch')
    # ax2.plot(epochs, validation_losses)

    # ax3.grid(True)
    # ax3.set_title('Training Losses')
    # ax3.set_xlabel('Epoch')
    # ax3.plot(epochs, training_losses)

    # plt.show()

    # model = checkpoint['model_state_dict']

    # print(checkpoint['train_loss'])
    # print(checkpoint['val_loss'])

    # model = checkpoint['model']

    # lrs = checkpoint['lrs']

    # fig, ax = plt.subplots()
    # ax.plot(lrs)    
    # ax.set(xlabel='epoch', ylabel='learning rate value')
    # fig.savefig(os.path.join(model_filepath, "lrs.png"))
    # print(f"{lrs = }")