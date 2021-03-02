#!/usr/bin/env python
# coding: utf-8

# In[17]:


import tensorflow as tf
import argparse
from PIL import Image
import time
import json
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()
import sys
import argparse

parser = argparse.ArgumentParser(description='Image classification')
image_path = parser.add_argument('image_path', type=str, help='Path of the image')
top_k = parser.add_argument('top_k', type=int, help='the top K possible class for the image')
args = parser.parse_args()
    
model_path = 'C:\\Users\\user\\Downloads\\mymodel.h5'
label_map = 'C:\\Users\\user\\Downloads\\label_map.json'

# top_k = 5
# image_path = 'C:\\Users\\user\\Downloads\\orange_dahlia.jpg'

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    return image

def predict(image_path, model_path, top_k):  
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    #print(processed_test_image.shape)
    probs = model.predict(processed_test_image)
    probs = probs[0].tolist()
    p,index = tf.math.top_k(probs, k=top_k)
    probs = p.numpy().tolist()
    classes = index.numpy().tolist()           
    return probs, classes

if __name__ == '__main__':
    #model = load_model(model_path)
    probs, classes = predict(image_path, model_path, top_k)
    #probs, classes = predict(args.image_path, model, args.top_k)
    with open(label_map, 'r') as f:
        class_names = json.load(f)
    for i in range(top_k):
        print(class_names[str(classes[i]+1)], ':', probs[i])


# In[ ]:




