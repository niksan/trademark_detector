import os
from PIL import Image
import image_conv_util

BASE_DIR = 'dataset'

base_list = os.listdir(BASE_DIR)

for group_folder in base_list:
  group_folder_dir = BASE_DIR + '/' + group_folder
  group_list = os.listdir(group_folder_dir)
  for set_folder in group_list:
    set_folder_dir = group_folder_dir + '/' + set_folder
    set_folder_list = os.listdir(set_folder_dir)
    for image_name in set_folder_list:
      print('.', end = '')
      image_path = set_folder_dir + '/' + image_name
      image = Image.open(image_path)
      layers = image.split()
      if len(layers) >= 4: # if alpha is present
        image = image_conv_util.convert_to_palette(image)
        image.convert('RGB').save(image_path)

print()