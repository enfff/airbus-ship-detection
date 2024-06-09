import numpy as np
import pandas as pd
import cv2
import torch
from utils import rl_decode
import gc
from tqdm import tqdm
import math

# Read the CSV file
df = pd.read_csv('train_ship_segmentations_v2.csv')

# To filter out stuff
# df = df[df['ImageId'] == '20ef0ae13.jpg']

groups = df.groupby('ImageId')

# use_me = groups.get_group('20ef0ae13.jpg')
# print(use_me)
# print(use_me["EncodedPixels"], type(use_me["EncodedPixels"]))
# print(use_me["ImageId"], type(use_me["ImageId"]))

# if use_me['EncodedPixels'].empty:
#   print("Empty")
# else:
#   print("Not empty")

# List to store the dictionaries
data = []

skipped = 0
iterations = 0

# For each group
for name, group in tqdm(groups):
    # Create a dictionary
    d = {'image_id': name, 'boxes': [], 'labels': []}

    # print(group, "\n\n\n")

    # if iterations >= 20:
    #   break

    # For each row in the group
    for index, row in group.iterrows():

      if isinstance(row['EncodedPixels'], float):
        # Non fa l'iter per trasformare EncodedPixels in bounding boxes (perché non ce ne sono!)
        # Quando stampavo row['EncodedPixels'] mi diceva sempre che era un float quando era vuoto. str altrimenti
        continue
    
      # print(type(row))
      # print(row)

      # print(type(row["EncodedPixels"]))
      # print(str(row["EncodedPixels"]))

      # if math.isnan(row['EncodedPixels']):
      #   print("it doesnt exists!")
      # else:
      #   print("it exist!")

      # print(type(row["EncodedPixels"]))
      # print(type(row["ImageId"]))

      # print(row["ImageId"], type(row["EncodedPixels"]))



      # if isinstance(row['EncodedPixels'], float):
      #   # It means the row is EMPTY. (Blame Pandas, not me!)
      #   if math.isnan(row['EncodedPixels']):
      #     # print("empty appended, ", d['image_id'])
      #     continue

      # print(row['EncodedPixels'], type(row['EncodedPixels']))
      
      ## From now on we assume it's a string

      # Decode the EncodedPixels to get the mask
      mask = rl_decode(row['EncodedPixels'], 768, 768)
      mask = (255*mask).byte().numpy()
      mask = cv2.resize(mask, (768, 768))

      # Find the contours in the mask and get the bounding box
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      x, y, w, h = cv2.boundingRect(contours[0])

      # Normalize the bounding box coordinates
      x1, y1 = x / 768, y / 768
      x2, y2 = (x + w) / 768, (y + h) / 768

      # Add the bounding box to the dictionary
      d['boxes'].append([x1, y1, x2, y2])
      d['labels'].append(1)  

    # Add the dictionary to the list
    data.append({
        'image_id': d['image_id'],
        'boxes': torch.tensor(d['boxes']),
        'labels': torch.tensor(d['labels'])
        })
    # print("final appended, ", d['image_id'])
    # print("appended, ", d['image_id'])

    iterations += 1
    if iterations % 10000 == 0:
        # Ogni 10k iterazioni svuota la RAM, altrimenti mi crasha il PC po
        gc.collect()
        # print("flushed")

print("Skipped images: ", skipped)
np.save('rcnn_targets.npy', data)
print("Saved file!")


print("Loading file...")
read_dictionary = np.load('rcnn_targets.npy',allow_pickle='TRUE')
print(len(read_dictionary))

# for el in read_dictionary:
#   print(el)
#   print("----")