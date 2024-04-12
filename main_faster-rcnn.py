import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
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


batch_size=32
img_dimensions = 224

# Normalize to the ImageNet mean and standard deviation
# Could calculate it for the cats/dogs data set, but the ImageNet
# values give acceptable results here.

# TODO RICONTROLLA to revise

img_train_transforms = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((img_dimensions, img_dimensions)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

img_validation_transforms = transforms.Compose([
    transforms.Resize((img_dimensions,img_dimensions)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

def img_label_transform(target):
    # TODO # sbrighiamocela dopo non si sa per ora, in caso leggi documentazione
    # target - dictionary containing
                # {
                #     "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
                #     "labels": [0, 0, ...]
                # }
    # return transformations on the boxes

    Warning('TODO')
    return None

# model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', weights=True) 

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def rl_decode(rl_str, height, length):
  mask = np.zeros(shape=(1,height,length))
  couples = rl_str.split()
  for i in range(0, len(couples)-1, 2):
    # print(i)
    el = int(couples[i])
    qty = int(couples[i+1])
    r,c = np.unravel_index(el,(height,length))
    for j in range(qty):
      mask[0, c+j-1, r-1] = 1

    # print(torch.Tensor(mask))
  return torch.Tensor(mask).reshape((768, 768)).gt(0)

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
    def __init__(self, file_list, targets, transforms = None):
        self.file_list = file_list
        self.targets = targets
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(self.root, folder_name))))
        # self.masks = list(sorted(os.listdir(os.path.join(self.root))))

    def __len__(self):
        self.filelength = len(self.file_list) 
        return self.filelength

    def __getitem__(self, idx):
        image = self.file_list[idx]     # numpy tensor
        label = self.targets[idx]       # dictionary {"boxes": , "label": }

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, self.targets[idx]

from sklearn.model_selection import train_test_split

DATASET_DIR = os.path.join("datasets", "airbus-ship-detection")
TRAIN_DIR = os.path.join("datasets", "airbus-ship-detection", "train_v2")
TEST_DIR = os.path.join("datasets", "airbus-ship-detection", "test_v2")

train_list = glob.glob(os.path.join(TRAIN_DIR,'*.jpg'))
train_list, val_list = train_test_split(train_list , test_size = 0.2)

train_data = ShipsDataset(train_list, transforms = img_train_transforms, targets=np.load('rcnn_targets.npy', allow_pickle='TRUE'))
# test_data = ShipsDataset(train_list, transforms = img_train_transforms)
val_data = ShipsDataset(val_list, transforms = img_validation_transforms, ) 

# TODO gestire shuffle
# Sostanzialmente targets deve essere splittato correttamente
# Magari togli opzione shuffle
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size = batch_size, shuffle = True)

print(len(train_data),len(train_loader))
print(len(val_data), len(val_loader))

print('arrivatooooo')

model_resnet50 = torchvision.models.detection.fasterrcnn_resnet50_fpn() # usa weights di default
# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
# La documentazione non è chiara sulla posizione dei punti per le ground-truth!

for name, param in model_resnet50.named_parameters():
      param.requires_grad = False

num_classes = 2 # ship, non-ship

# model_resnet18.fc = nn.Sequential(nn.Linear(model_resnet18.fc.in_features,128),
#                                   nn.Linear(128, num_classes))

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device="gpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item() * inputs.size(0)

            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model_resnet50.to(device)
torch.compile(model)
optimizer = optim.Adam(params = model.parameters(),lr =0.01)
criterion = nn.CrossEntropyLoss()
train(model, optimizer, torch.nn.CrossEntropyLoss(), train_loader, val_loader, epochs=5, device=device)