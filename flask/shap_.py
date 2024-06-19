import shap
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os, sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer

def load_custom_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    sys.path.append(model_path)

    get_custom_objects().update({
        'ConvKernalInitializer': ConvKernalInitializer,
        'Swish': Swish,
        'DropConnect': DropConnect
    })

    model = tf.keras.models.load_model(model_path)
    return model

model_path = 'static/my_models/Gender_Prediction.h5' 
model = load_custom_model(model_path)
model.summary()


model.output_names
print(model.output_names)

model.output_names[0] # age estimation
print(model.output_names[0])

# model.output_names[1] # sex estimation
# print(model.output_names[1])

# #Create separate models for each output you want to explain
# #AGE
# model_layer1 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('prediction_layer').output)
# print(model_layer1)
# #Gender
model_layer2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('prediction_layer2').output)
print(model_layer2)


def process_input(image_path):
    background_data = []

    image = load_img(image_path, target_size=(224, 224))
    preprocessed_image = img_to_array(image) / 255.0
    background_data.append(preprocessed_image)

    return np.array(background_data)

def process_bg(images_directory):
    background_data = []
    image_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        try:
            image = load_img(image_path, target_size=(224, 224))
            preprocessed_image = img_to_array(image) / 255.0
            background_data.append(preprocessed_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return np.array(background_data)



image_path = 'static/uploads/images/left_opg_2.png'
background_upload = process_input(image_path)
# Convert background data to numpy array
background_upload_np = np.array(background_upload)
print(background_upload_np.shape)

# Define the base path for your images
images_base_path = "images_bg"
# Create background data using the process_input function
background_train = process_bg(images_base_path)
# Convert background data to numpy array
background_train_np = np.array(background_train)


###Gender
explainer_layer2 = shap.GradientExplainer(model_layer2, background_train_np)

shap_values_left_opg_2 = explainer_layer2.shap_values(background_upload_np)
shap.image_plot(shap_values_left_opg_2, background_upload_np)

