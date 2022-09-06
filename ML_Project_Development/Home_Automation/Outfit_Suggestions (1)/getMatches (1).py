#https://github.com/google/making_with_ml/blob/master/instafashion/scripts/getMatches.ipynb
from pyvisionproductsearch import ProductSearch, ProductCategories
from google.cloud import storage
from google.cloud import firestore
import pandas as pd

from google.cloud import vision
from google.cloud.vision_v1 import types
from utils import detectLabels, detectObjects
import io
from tqdm.notebook import tqdm
import os
from dotenv import load_dotenv
load_dotenv()
## incomplete

# Fill these out with your own values
# GCP config
GCP_PROJECTID="YOUR_PROJECT_ID"
BUCKET="YOUR_BUCKET"
CREDS="key.json"
PRODUCT_SET="YOUR_PRODUCT_SET"
INSPO_BUCKET = "YOUR_INSPO_PIC_BUCKET"
# If your inspiration pictures are in a subfolder, list it here:
INSPO_SUBFOLDER = "YOUR_SUBFOLDER_NAME"

# To use this notebook, make a copy of .env_template --> .env and fill out the fields!
ps = ProductSearch(GCP_PROJECTID, CREDS, BUCKET)
productSet = ps.getProductSet(PRODUCT_SET)

matchGroups = [("skirt", "miniskirt"),
               ("jeans", "pants"),
               ("shorts"),
               ("jacket", "vest", "outerwear", "coat", "suit"),
               ("top", "shirt"),
               ("dress"),
               ("swimwear", "underpants"),
               ("footwear", "sandal", "boot", "high heels"),
               ("handbag", "suitcase", "satchel", "backpack", "briefcase"),
               ("sunglasses", "glasses"),
               ("bracelet"),
               ("scarf", "bowtie", "tie"),
               ("earrings"),
               ("necklace"),
               ("sock"),
               ("hat", "cowboy hat", "straw hat", "fedora", "sun hat", "sombrero")]

    for group in matchGroups:
        if label1.lower() in group and label2.lower() in group:
            return True
    return False
