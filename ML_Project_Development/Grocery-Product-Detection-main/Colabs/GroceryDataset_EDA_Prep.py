# %% markdown
# <a href="https://colab.research.google.com/github/sayakpaul/Grocery-Product-Detection/blob/main/Colabs/GroceryDataset_EDA_Prep.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% markdown
# This notebook prepares the [Grocery Dataset](https://github.com/gulvarol/grocerydataset) to train an object detection model to detect products from a store shelf image.
# %% markdown
# ## Gather data sources
# %% codecell
!wget -q https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz
!tar xf ShelfImages.tar.gz
# %% codecell
!wget -q https://github.com/gulvarol/grocerydataset/releases/download/1.0/GroceryDataset_part2.tar.gz
!tar xf GroceryDataset_part2.tar.gz
# %% codecell
!ls -lh ShelfImages | head -10
!ls -lh ShelfImages/train | head -10
!ls -lh ShelfImages/test | head -10
# %% markdown
# From [here](https://github.com/gulvarol/grocerydataset#shelfimages) we can get a sense of how an individual image is named:
#
# ```
# "C<c>_P<p>_N<n>_S<s>_<i>.JPG"
#     where
#         <c> := camera id (1: iPhone5S, 2: iPhone4, 3: Sony Cybershot, 4: Nikon Coolpix)
#         <p> := planogram id
#         <n> := the rank of the top shelf on the image according to the planogram
#         <s> := number of shelves on the image
#         <i> := copy number
# ```
# %% markdown
# ## Imports
# %% codecell
from imutils import paths
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %% markdown
# ## Visualize the raw images
# %% codecell
train_images = list(paths.list_images(path))
plt.figure(figsize=(15, 15))
for i, image  in enumerate(train_images[:4]):
    img = cv2.imread(image)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray,150,100,apertureSize = 3)
    #display image
    plt.figure(1)
    plt.imshow(img)
    plt.imshow(img)
    plt.show()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,150,100,apertureSize = 3)
    # ax = plt.subplot(3, 3, i + 1)
    # plt.figure(1)
    # plt.imshow(image)
    # plt.axis("off")
    # pause(10)
# %% markdown
# A question that gets raised here - how do I map these images to their detection annotations? From [here](https://github.com/gulvarol/grocerydataset#annotationtxt) we have the following information (which is summarized in [`annotations.csv`](https://github.com/gulvarol/grocerydataset/blob/master/annotations.csv)) -
# %% markdown
# ```
# <shelf image name> <n> <x_1> <y_1> <w_1> <h_1> <b_1> <x_2> <y_2> <w_2> <h_2> <b_2> ... <x_n> <y_n> <w_n> <h_n> <b_n>
#     where
#         <shelf image name>   := shelf image name
#         <n>                  := number of product on the shelf image
#         <x_i>                := x-coordinate of the i'th product image
#         <y_i>                := y-coordinate of the i'th product image
#         <w_i>                := width of the i'th product image
#         <h_i>                := height of the i'th product image
#         <b_i>                := brand of the i'th product image
# ```
# %% markdown
# ## Visualize bbox annotations
# %% codecell
cols = ["image_name", "x_i", "y_i", "w_i", "h_i", "b_i"]
master_df = pd.read_csv("https://raw.githubusercontent.com/gulvarol/grocerydataset/master/annotations.csv",
                        names=cols)
master_df.head()
# %% codecell
# How many unique brands? (0 stands for "other" class)
master_df["b_i"].unique()
# %% markdown
# Let's visualize a few images with their respective annotations. We will write a small utility for this purpose.
# %% codecell
def vis_annotations(image_path: str, coordinate_list: List[List[int]],
                    color: str="blue") -> None:
    """Converts bounding box to matplotlib format, imposes it on the
    provided image and then displays the plot."""
    image = plt.imread(image_path)
    fig = plt.imshow(image)
    for i in range(len(coordinate_list)):
        bbox = coordinate_list[i]
        fig.axes.add_patch(plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
            fill=False, edgecolor=color, linewidth=2))
    plt.show()
# %% markdown
# Let's start with the first image from the dataframe above. Its absolute path is present in `train_images`.
# %% codecell
train_images[0]
# %% codecell
coordinate_columns = master_df.columns[1:-1]
coordinate_list = master_df[master_df["image_name"]=="C1_P01_N1_S2_1.JPG"][coordinate_columns]
coordinate_list = coordinate_list.values.tolist()
vis_annotations("ShelfImages/train/C1_P01_N1_S2_1.JPG", coordinate_list)
# %% markdown
# ## Splitting into train and test dataframes
# %% markdown
# We already have the train and test images segregated. We can use that information to split `master_df` into train and test dataframes.
# %% codecell
# Grab the image names belonging to the train and test sets
train_images = list(paths.list_images("ShelfImages/train"))
test_images = list(paths.list_images("ShelfImages/test"))
train_image_names = [image_path.split("/")[-1] for image_path in train_images]
test_image_names = [image_path.split("/")[-1] for image_path in test_images]
print(len(train_image_names), len(test_image_names))
# %% codecell
# Create two different dataframes from train and test sets
train_df = master_df[master_df["image_name"].isin(train_image_names)]
test_df = master_df[~master_df["image_name"].isin(train_image_names)]
print(len(np.unique(train_df["image_name"])), len(np.unique(test_df["image_name"])))
# %% codecell
# Let's turn the image names into absolute paths
train_df["image_name"] = train_df["image_name"].map(lambda x: "ShelfImages/train/" + x)
test_df["image_name"] = test_df["image_name"].map(lambda x: "ShelfImages/test/" + x)
# %% codecell
# Preview
train_df.head()
# %% markdown
# ## Setup TFOD API
# %% codecell
%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)

!git clone https://github.com/tensorflow/models.git

% cd models/research
!pip install --upgrade pip
# Compile protos.
!protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
!cp object_detection/packages/tf1/setup.py .
!python -m pip install --use-feature=2020-resolver .
# %% markdown
# ## Generating TFRecords
#
# We will be using the [TFOD API](https://github.com/tensorflow/models/tree/master/research/object_detection) to train the detection model. The API expects TFRecords as its inputs. So, before we start the model training process we first need to represent our dataset in terms of TFRecords only.
#
# A single example inside the TFRecords should have the following entries -
# * filename,
# * width,
# * height,
# * class,
# * xmin,
# * ymin,
# * xmax,
# * ymax
#
# **Note**: Although the official repository of the Grocery Dataset mentions that there are width and height coordinates I don't they are so. I think they are bottom-right coordinates. Also, TFOD API reserves 0 for the background label. Therefore, we will add 1 to all the brand IDs.
#
# Given all of these, let's start by writing a utility to create our dataframes in the expected format.
# %% codecell
def prepare_df(original_df):
    df = pd.DataFrame()
    df["filename"] = original_df["image_name"]
    df["wdith"] = (original_df["x_i"] - original_df["w_i"]).astype("int")
    df["height"] = (original_df["y_i"] - original_df["h_i"]).astype("int")
    df["class"] = (original_df["b_i"] + 1).astype("int")
    df["xmin"] = (original_df["x_i"]).astype("int")
    df["ymin"] = (original_df["y_i"]).astype("int")
    df["xmax"] = (original_df["w_i"]).astype("int")
    df["ymax"] = (original_df["h_i"]).astype("int")
    return df
# %% codecell
new_train_df = prepare_df(train_df)
new_test_df = prepare_df(test_df)
print(len(np.unique(new_train_df["filename"])), len(np.unique(new_test_df["filename"])))
# %% codecell
# Serialize these dataframes
new_train_df.to_csv("train.csv", index=False)
new_test_df.to_csv("test.csv", index=False)
# %% codecell
!ls -lh *.csv
# %% markdown
# I wrote this [utility script](https://gist.github.com/sayakpaul/d82a43c03089a8abfb5b042ee89eeb32) to help us generate the TFRecords.
# %% codecell
!wget -q -O generate_tfrecord.py https://gist.githubusercontent.com/sayakpaul/d82a43c03089a8abfb5b042ee89eeb32/raw/fee76357235803c6a0d2d8859e72542c7a916340/generate_tfrecord.py
# %% codecell
!python generate_tfrecord.py \
    --csv_input=/content/models/research/train.csv \
    --output_path=/content/train.record
!python generate_tfrecord.py \
    --csv_input=/content/models/research/test.csv \
    --output_path=/content/test.record

!ls -lh *.record
# %% markdown
# ## Generate `.pbtxt`
#
# We need to generate a `.pbtxt` file that defines a mapping between our classes and integers. In our case, the classes are already integers. But we still need this file for the TFOD API to operate.
# %% codecell
classes = new_train_df["class"].unique()
label_encodings = {}
for cls in classes:
    label_encodings[str(cls)] = int(cls)

f = open("/content/label_map.pbtxt", "w")

for (k, v) in label_encodings.items():
    item = ("item {\n"
            "\tid: " + str(v) + "\n"
            "\tname: '" + k + "'\n"
            "}\n")
    f.write(item)

f.close()

!cat /content/label_map.pbtxt
# %% markdown
# ## Moving files to Google Drive for later usage
# %% codecell
from google.colab import drive
drive.mount('/content/drive')
# %% codecell
!mkdir /content/drive/MyDrive/product-detection
!cp -r *.record /content/drive/MyDrive/product-detection
!cp -r *.pbtxt /content/drive/MyDrive/product-detection
