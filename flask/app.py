from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
import os
import torch
import subprocess
import json
import shap
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from langdetect import detect
import sys
import json
from joblib import load
from transformers import BertTokenizer, TFBertModel



app = Flask(__name__)

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model_7_23 = load_model('../Webapp/templates/26_Multi_1e-6_250_Unfreeze.h5')
model_7_14 = load_model('../Webapp/templates/36_Multi_1e-5_500_Unfreeze.h5')
model_15_23 = load_model('../Webapp/templates/25_Multi_1e-6_500_Unfreeze.h5')

# Load your models outside of the request to save loading time
random_forest_model = load('../Webapp/templates/logreg_classifier1.joblib')  # Adjust path as needed
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
