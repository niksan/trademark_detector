import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only errors

# tf.config.set_visible_devices([], 'GPU') # for force using CPU instead of GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = 'predictable'
TRAINED_MODEL_FILENAME = 'trained_model.h5'

model = tf.keras.models.load_model(TRAINED_MODEL_FILENAME)

images_list = os.listdir(BASE_DIR)

for image_name in images_list:
  path = BASE_DIR + '/' + image_name
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  if classes[0]>0.5:
    print(image_name + " is a nike")
  else:
    print(image_name + " is a disney")