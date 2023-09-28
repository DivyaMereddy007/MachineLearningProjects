#https://www.kaggle.com/code/shivamb/cnn-architectures-vgg-resnet-inception-tl
#Main topics resnet intro and trasnfer learning to predict fruits type
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import os

img1 = "../input/dogs-vs-cats-redux-kernels-edition/train/cat.11679.jpg"
img2 = "../input/dogs-vs-cats-redux-kernels-edition/train/dog.2811.jpg"
img3 = "../input/flowers-recognition/flowers/flowers/sunflower/7791014076_07a897cb85_n.jpg"
img4 = "../input/fruits/fruits-360_dataset/fruits-360/Training/Banana/254_100.jpg"
imgs = [img1, img2, img3, img4]

def _load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def _get_predictions(_model):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 40)
    for i in range(4):
        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()

    f, axes = plt.subplots(1, 4)
    f.set_size_inches(80, 20)
    for i,img_path in enumerate(imgs):
        img = _load_image(img_path)
        preds  = decode_predictions(_model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=55)
        f.tight_layout()
