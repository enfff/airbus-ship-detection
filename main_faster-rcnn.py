from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms.functional import rotate
from torchvision.utils import draw_segmentation_masks
from torchvision.ops import masks_to_boxes
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import tv_tensors
import logging
from datetime import datetime

device= 'cpu'
batch_size=32
img_dimensions = 224

create_log_file = True

if create_log_file:
    log_filename = os.getlogin() + "_" + datetime.now().strftime("%m-%d_%H.%M.%S")
    log_filename = os.path.join("logs", log_filename + ".txt")
    # print(log_filename)

    logging.basicConfig(filename=log_filename,
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        datefmt='%m-%d %H:%M:%S')
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        datefmt='%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

# Normalize to the ImageNet mean and standard deviation
# Could calculate it for the cats/dogs data set, but the ImageNet
# values give acceptable results here.

# TODO RICONTROLLA to revise

img_train_transforms = transforms.Compose([
    v2.RandomRotation(50),
    v2.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Resize((img_dimensions, img_dimensions), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

img_validation_transforms = transforms.Compose([
    transforms.Resize((img_dimensions,img_dimensions), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

""" def img_label_transform(target):
    # TODO # sbrighiamocela dopo non si sa per ora, in caso leggi documentazione
    # target - dictionary containing
                # {
                #     "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
                #     "labels": [0, 0, ...]
                # }
    # return transformations on the boxes

    Warning('TODO')
    return None """

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def show(imgs, rotation=None):

    if rotation:
          imgs = rotate(imgs, rotation)

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class ShipsDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, targets, transforms = None, target_transforms = None):
        self.file_list = file_list
        self.targets = targets
        self.transform = transforms

    def __len__(self):
        self.filelength = len(self.file_list) 
        return self.filelength

    def __getitem__(self, idx):
        image = read_image(self.file_list[idx])    # numpy tensor
        # try:
        label = self.targets[idx]       # dictionary {"boxes": , "label": }
        label['boxes'] = torch.Tensor(label['boxes'])
        label['labels'] = torch.Tensor(label['labels']).to(dtype=torch.int64).reshape((-1,))
        # except IndexError as e:
        #     Warning(f'Errore con {idx = }')
        #     plt.imshow(image.permute(1, 2, 0))
        #     plt.show()
            
        if self.transform:
            image, label = self.transform((image,label))
        
        return image, label
        
from sklearn.model_selection import train_test_split

DATASET_DIR = os.path.join("datasets", "airbus-ship-detection")
TRAIN_DIR = os.path.join("datasets", "airbus-ship-detection", "train_v2")
TEST_DIR = os.path.join("datasets", "airbus-ship-detection", "test_v2")

train_list = glob.glob(os.path.join(TRAIN_DIR,'*.jpg'))
train_list, val_list = train_test_split(train_list , test_size = 0.2)

train_data = ShipsDataset(train_list, transforms = img_train_transforms, targets=np.load('rcnn_targets.npy', allow_pickle='TRUE'))
# test_data = ShipsDataset(train_list, transforms = img_train_transforms)
val_data = ShipsDataset(val_list, transforms = img_validation_transforms,targets=np.load('rcnn_targets.npy', allow_pickle='TRUE') ) 

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, collate_fn=lambda x: x)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size = batch_size, shuffle = True, collate_fn=lambda x: x)

print(len(train_data),len(train_loader))
print(len(val_data), len(val_loader))

model_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT') 
# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
# La documentazione non è chiara sulla posizione dei punti per le ground-truth!
# /Users/ludovicamazzucco/Library/Python/3.9/lib/python/site-packages/torchvision/models/detection/generalized_rcnn.py"

## STEP 1. freeze backbone layers, add final layers and train the network

for name, param in model_rcnn.named_parameters():
      param.requires_grad = False

num_classes = 2 # background, ship

in_features = model_rcnn.roi_heads.box_predictor.cls_score.in_features
model_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# what last layer of model_rcnn is like
""" (roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
    (box_head): TwoMLPHead(
      (fc6): Linear(in_features=12544, out_features=1024, bias=True)
      (fc7): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=91, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
    ) """

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=1, device= device):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for i, batch in enumerate(train_loader):
            logger.info("batch " + str(i))
            print(batch, i) # this should be unneccessary
            optimizer.zero_grad()
            #inputs, targets = batch
            """ inputs = [img for i,el in enumerate(batch)]     
            targets = [lab for img,lab in batch] """

            # filtering out empty images (model does not accept empty targets)
            inputs = []
            targets = []
            for el in batch:       # el = (image,labels)
                if el[1]['boxes'].size()[0] != 0:
                    inputs.append(el[0])
                    targets.append(el[1])
            
           # inputs = inputs.to(device)
           # targets = targets.to(device)
            output = model(inputs,targets)  # NOTE: output is a dict with already computed losses within!

            """ EXAMPLE :
            {'loss_classifier': tensor(1.0206, grad_fn=<NllLossBackward0>), 
             'loss_box_reg': tensor(0.0071, grad_fn=<DivBackward0>), 
             'loss_objectness': tensor(1.8541), 'loss_rpn_box_reg': tensor(1.8591)} """
            
            loss = sum(loss for loss in output.values())
            #train_loss_list.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * len(inputs)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs = [img for img,lab in batch]
            targets = [lab for img,lab in batch]
           # inputs = inputs.to(device)
            output = model(inputs)
           # targets = targets.to(device)
            loss = sum(loss for loss in output.values())
            valid_loss += loss.data.item() * len(inputs)

            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))

""" if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
 """
model = model_rcnn.to(device)
torch.compile(model)
optimizer = optim.Adam(params = model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train(model, optimizer, torch.nn.CrossEntropyLoss(), train_loader, val_loader, epochs=1 , device=device)
torch.save(model.state_dict(), 'model_state_dict')