import pandas as pd
import cv2
import torch
from utils import rl_decode
from tqdm import tqdm
import glob

# Read the CSV file
df = pd.read_csv('datasets/airbus-ship-detection/train_ship_segmentations_v2.csv')
# print(df.head(20))

grouped_df = df.groupby('ImageId')['EncodedPixels'].apply(lambda x: list(x.fillna(''))).reset_index()

# Limit the DataFrame to the first 10 rows
# first_10_rows = grouped_df.head(30)

# Iterate over the rows

filenumber = 0
new_targets = []
mod = 20_000

for index, row in tqdm(grouped_df.iterrows()):

  if index % mod == 0 and index > 0: # Save the file every 20k indices
      torch.save(new_targets, 'rcnn_targets' + str(filenumber) + '.pt')
      print("\ncreated new .pt file. last index: ", index, "\n")
      new_targets = []
      filenumber += 1

  image_id = row['ImageId']
  encoded_pixels = row['EncodedPixels']

  tmp_dict = {
      "boxes": torch.FloatTensor([]),
      "labels": torch.LongTensor([]),
      "image_id": image_id
    }

  if encoded_pixels == ['']:
      new_targets.append(tmp_dict)
      continue
  else:
    for label in encoded_pixels:
      mask = rl_decode(label, 768, 768)
      mask = (255*mask).byte().numpy()
      mask = cv2.resize(mask, (768, 768))

      contours, hierarchy = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      x, y, w, h = cv2.boundingRect(contours[0])

      tmp_dict["boxes"] = torch.cat([tmp_dict["boxes"], torch.FloatTensor([[x, y, x+w, y+h]])])
      tmp_dict["labels"] = torch.cat([tmp_dict["labels"], torch.LongTensor([1])])  # only one class: ship
    
    new_targets.append(tmp_dict)
      

# Last filesave
torch.save(new_targets, 'rcnn_targets' + str(filenumber) + '.pt')
print("\ncreated last .pt file")
new_targets = []



### Merge files
print("Merging files...")
# Get a list of all files that match the pattern
files = glob.glob('rcnn_targets*.pt')
files = sorted(files)

all_data = [] # Initialize an empty list to hold all the data

# Loop over the files and load each one
for file in tqdm(files):
    data = torch.load(file)
    all_data.extend(data)
    
# Save the combined data to a new file
torch.save(all_data, 'rcnn_targets.pt')
print("Merged files and saved to 'rcnn_targets.pt'")



### Testing the results

data = torch.load('rcnn_targets.pt')

print(type(data)) # <class 'list'>
print(type(data[3]['labels'])) # <class 'torch.Tensor'>
print(data[3]['boxes'])
print(data[3]['labels'])
print(data[3]['image_id'])

print(data[3].keys())