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

DATASET_NAME = "airbus-ship-detection"
DATASET_LABEL_PATH = os.path.join("datasets", DATASET_NAME, "labels")

if os.path.exists(DATASET_LABEL_PATH):
   shutil.rmtree(DATASET_LABEL_PATH)
else:
    os.makedirs(DATASET_LABEL_PATH)

for index, row in targets.iterrows():
    image_id = row['ImageId']
    label = row['EncodedPixels']

    box = None

    if not pd.isna(label):
        mask = rl_decode(label, 768, 768)
        mask = (255*mask).byte().numpy()
        mask = cv2.resize(mask, (768, 768))

        contours, hierarchy = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        # x, y top left coordinate of the rectangle
        
        x = x + w/2 # to get the x coordinate of the center
        y = y + h/2 # to get the y coordinate of the center

        x = x / 768
        y = y / 768
        w = w / 768
        h = h / 768

        # I did this because each row of yolov5 labels is: class x_center y_center width height
        # print(x, y, w, h)

        image_id = image_id.replace('jpg', 'txt') # rimuove estensione

        filepath = os.path.join(DATASET_LABEL_PATH, image_id) 

        with open(filepath, "a") as f:
            # 0 corresponsts to class ship
            # print(f"Writing in {filepath} 0 {x} {y} {w} {h} ")
            f.write(f"0 {x} {y} {w} {h}\n")