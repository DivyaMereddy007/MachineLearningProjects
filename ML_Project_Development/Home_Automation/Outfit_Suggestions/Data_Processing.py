from google.api_core.protobuf_helpers import get_messages
from google.cloud import vision

import sys, types, os;
from google.cloud.vision_v1 import types

import pandas as pd
import numpy as np
import googlemaps
from datetime import datetime
import pandas as pd
import sys

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\divya.mereddy\\OneDrive - Nolan Transportation Group\\Documents\\Divya\\MachineLearningProjects-master\\Home_Automation\\Outfit_Suggestions\\cobalt-upgrade-335321-c19b741b41c0.json"

client = vision.ImageAnnotatorClient()

import io

path = 'C:\\Users\\divya.mereddy\\OneDrive - Nolan Transportation Group\\Documents\\Divya\\MachineLearningProjects-master\\Home_Automation\\Outfit_Suggestions\\Data\\Picture\\Training\\Sample.jpg'
with io.open(path, 'rb') as image_file:
        content = image_file.read()

from __future__ import print_function
from google.cloud import vision

image_uri = 'gs://cloud-samples-data/vision/using_curl/shanghai.jpeg'
image_uri = 'gs://divyabucket1/Processed_Data/Tops/10.PNG'

client = vision.ImageAnnotatorClient()
image = vision.Image()
image.source.image_uri = image_uri

objects = client.object_localization(image=image).localized_object_annotations
type(objects[0])
pd.read_json(objects,orient='index')
print('Number of objects found: {}'.format(len(objects)))
#1]
object_name=objects[0].name

for object_ in objects:

        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

#client.safe_search_detection(image=image)

response = client.label_detection(image=image)


Street_Fashion_Flag=0
Denim_Flag=0
print('Labels (and confidence score):')
print('=' * 30)
for label in response.label_annotations:
    print(label.description, '(%.2f%%)' % (label.score*100.))
    if(label.description=='Street Fashion'):
        Street_Fashion_Flag=1
    if(label.description=='Denim'):
        Denim_Flag=1

#

response = client.image_properties(image=image)
props = response.image_properties_annotation
print('Properties:')

def hex_format(RGB):
		'''Returns color in hex format'''
		return '#{:02X}{:02X}{:02X}'.format(red,green,blue)
color=props.dominant_colors.colors[0]
RGB=(color.color.red,color.color.green,color.color.blue)
for color in props.dominant_colors.colors:
    print('fraction: {}'.format(color.pixel_fraction))
    print('\tr: {}'.format(color.color.red))
    print('\tg: {}'.format(color.color.green))
    print('\tb: {}'.format(color.color.blue))
    RGB=hex_format(mcolors.to_rgb('red'))
    RGB=mcolors.rgb_to_hsv((color.color.red,color.color.green,color.color.blue))
    print('\ta: {}'.format(color.color.alpha))

if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))

df.iloc[len(df.index)]=
#################


client = vision.ImageAnnotatorClient()
image = vision.Image()
image.source.image_uri = "gs://divyabucket1/Sample.jpg"

response = client.label_detection(image=image)

print('Labels (and confidence score):')
print('=' * 30)
for label in response.label_annotations:
    print(label.description, '(%.2f%%)' % (label.score*100.))

#########################

from utils import detectLabels, detectObjects
image_source = types.ImageSource(image_uri=path)
labels = detectLabels(image=image)
  # Only save images that have the label "Fashion"
if any([x.description == "Fashion" for x in labels]):
    fashionPics.append(uri)
