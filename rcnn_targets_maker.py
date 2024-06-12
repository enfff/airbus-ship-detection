import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
import glob

# During training, the model expects both the input tensors and a targets (list of dictionary), containing:
# boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
# labels (Int64Tensor[N]): the class label for each ground-truth box

# targets = [
#     {
#         'boxes': torch.FloatTensor([[x1_1, y1_1, x2_1, y2_1], [x1_2, y1_2, x2_2, y2_2], ...]),  # Ground-truth bounding boxes for each object
#         'labels': torch.tensor([label_1, label_2, ...])  # Class labels for each object
#         'image_id': image_id.jpg   
# },
#     {
#         'boxes': torch.FloatTensor([[x1_1, y1_1, x2_1, y2_1], [x1_2, y1_2, x2_2, y2_2], ...]),  # Ground-truth bounding boxes for each object
#         'labels': torch.tensor([label_1, label_2, ...])  # Class labels for each object
#         'image_id': image_id.jpg       
# },
#     ...
# ]

# This script reads and convert the data from the csv file to the format above.
# The data is saved in multiple .npy files, each containing a list of dictionaries.
# The final step is to merge all the files into a single .npy file.
# I had to do that because otherwise we would run out of RAM memory.

new_targets = []
skip_empty = False
last_image_id = None

targets = pd.read_csv("train_ship_segmentations_v2.csv")


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


tmp_dict = {      # dict to append to new_targets
  "boxes": torch.FloatTensor([]),
  "labels": torch.IntTensor([]),
  "image_id": None,
}

iter = 0
mod = 20_000

for index, row in tqdm(targets.iterrows()):

    image_id = row['ImageId']
    label = row['EncodedPixels']

    if skip_empty:        # Skips images with empty labels
      if pd.isna(label):
        continue

    if last_image_id != image_id: # generates new dict for new image
      last_image_id = image_id

      new_targets.append(tmp_dict)  # append to new_targets

      if int(index / mod) > 0:
        np.save(f'rcnn_targets{iter}.npy', new_targets)
        new_targets = []
        iter += 1
        mod += 20_000
        print("\ncreated new .npy file. last index: ", index, "\n")

      # reset to restart
      tmp_dict = {
        "boxes": torch.FloatTensor([]),
        "labels": torch.IntTensor([]),
        "image_id": image_id
      }

    if not pd.isna(label):
        mask = rl_decode(label, 768, 768)
        mask = (255*mask).byte().numpy()
        mask = cv2.resize(mask, (768, 768))

        contours, hierarchy = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        tmp_dict["boxes"] = torch.cat([tmp_dict["boxes"], torch.FloatTensor([[x, y, x+w, y+h]])])
        tmp_dict["labels"] = torch.cat([tmp_dict["labels"], torch.Tensor([1])])  # only one class: ship

# saves the last one
np.save(f'rcnn_targets{iter}.npy', new_targets)
new_targets = []
iter += 1
print("\nsaved last .npy file")


### Merge files
print("Merging files...")
# Get a list of all files that match the pattern
files = glob.glob('rcnn_targets*.npy')
files = sorted(files)

# Initialize an empty list to hold all the data
all_data = []

# Loop over the files and load each one
for file in tqdm(files):
    data = np.load(file, allow_pickle=True)

    if file == files[0]:
        data = data[1:]
        # Quick and dirty solution. Skips the first entry, because its
        # image_id is set to None due to bad programming :).

    all_data.extend(data)

# Save the combined data to a new file
np.save('rcnn_targets.npy', all_data.tolist())

del all_data
import gc
gc.collect()

# Load the combined data
data = np.load('rcnn_targets.npy', allow_pickle=True)