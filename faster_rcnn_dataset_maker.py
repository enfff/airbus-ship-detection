import numpy as np
import pandas as pd
import torch
import cv2
import os
import matplotlib.pyplot as plt
import shutil

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

targets = pd.read_csv("train_ship_segmentations_v2.csv")

new_targets = []

last_image_id = None

tmp_dict = {      # dict to append to new_targets
  "boxes": [],
  "labels": [],
}

for index, row in targets.iterrows():
    image_id = row['ImageId']
    label = row['EncodedPixels']

    if last_image_id != image_id:
      last_image_id = image_id

      new_targets.append(tmp_dict)
      # print(f"New dict appended: {tmp_dict = }")

      # reset to restart
      tmp_dict = {
        "boxes": [],
        "labels": [],
      }


    box = None

    if not pd.isna(label):
        mask = rl_decode(label, 768, 768)
        mask = (255*mask).byte().numpy()
        mask = cv2.resize(mask, (768, 768))

        contours, hierarchy = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        # x, y top left coordinate of the rectangle
        
        x1, y1 = x, y           # top-left
        x2, y2 = x + w, y + h   # bottom-right <- assumption! the documentation isn't too precise!
                                # https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

        x1 = x1 / 768
        x2 = x2 / 768
        y1 = y1 / 768
        y2 = y2 / 768

        w = w / 768
        h = h / 768

        assert(x1 <= 1 and x1 >= 0)
        assert(x2 <= 1 and x2 >= 0)
        assert(y1 <= 1 and y1 >= 0)
        assert(y2 <= 1 and y2 >= 0)
        assert(w <= 1 and w>= 0)
        assert(h <= 1 and h>= 0)

        tmp_dict["boxes"] += [[x1, y1, x2, y2]]
        tmp_dict["labels"] += [[1]]                   # only one class: ship

        # During training, the model expects both the input tensors and a targets (list of dictionary), containing:
        # boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        # labels (Int64Tensor[N]): the class label for each ground-truth box

        # targets = [
        #     {
        #         'boxes': torch.tensor([[x1_1, y1_1, x2_1, y2_1], [x1_2, y1_2, x2_2, y2_2], ...]),  # Ground-truth bounding boxes for each object
        #         'labels': torch.tensor([label_1, label_2, ...])  # Class labels for each object
        #     },
        #     {
        #         'boxes': torch.tensor([[x1_1, y1_1, x2_1, y2_1], [x1_2, y1_2, x2_2, y2_2], ...]),  # Ground-truth bounding boxes for each object
        #         'labels': torch.tensor([label_1, label_2, ...])  # Class labels for each object
        #     },
        #     ...
        # ]

np.save('rcnn_targets.npy', new_targets)

read_dictionary = np.load('rcnn_targets.npy',allow_pickle='TRUE')
print(read_dictionary)