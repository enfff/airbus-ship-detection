import numpy as np
import pandas as pd
import torch
import cv2
import csv

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
new_rows = []

with open('train_ship_bounding_boxes_v2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["ImageId", "BoundingBox"]
    writer.writerow(field)

    for index, row in targets.iterrows():
        image_id = row['ImageId']
        label = row['EncodedPixels']

        box = None

        if not pd.isna(label):
            mask = rl_decode(label, 768, 768)
            mask = (255*mask).byte().numpy()
            mask = cv2.resize(mask, (768, 768))

            contours, hierarchy = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])

            if rect[1][1] > rect[1][0]:
                angle = 90-rect[2]
            else:
                angle = -rect[2]

            box = cv2.boxPoints(rect)
            box = np.intp(box)

        writer.writerow([image_id, box])