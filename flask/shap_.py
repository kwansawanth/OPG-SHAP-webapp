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
from PIL import Image


app = {'config': {'FOLDER_SHAP2ND': 'static/uploads/shap2nd'}}

def load_custom_model(model_select):
    model_path = model_select
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

def process_input(left_image_output_path):
    image_path = left_image_output_path
    background_data = []

    image = load_img(image_path, target_size=(224, 224))
    preprocessed_image = img_to_array(image) / 255.0
    background_data.append(preprocessed_image)

    return np.array(background_data)


def main(left_image_output_path, model_select, background_train_np):
    
    #-------------------LOAD MODEL----------------------------------
    model = load_custom_model(model_select)
    model.summary()

    model.output_names
    print(model.output_names)

    model.output_names # age estimation
    print(model.output_names)
    
    model_name = model_select.split('/')[-1]
    input_model_layer = model.output_names[0]
    age_model_layer = model.output_names[0]
    gender_model_layer = model.output_names[1]
    
    
    # model.output_names[1] # sex estimation
    # print(model.output_names[1])

    # #Create separate models for each output you want to explain
    # #AGE
    # model_layer1 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('prediction_layer').output)
    # print(model_layer1)
    # #Gender
    
    #-------------------ADD CONDITION TO USE LAYER---------------------
    if model_name == 'Age_estimation.h5':  # Age estimation
        model_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer(age_model_layer).output)
    elif model_name == 'Gender_Prediction.h5':  # Gender prediction
        model_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer(gender_model_layer).output)
    else:
        model_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer(input_model_layer).output)
        
    print(model_layer)

    #-------------------LOAD BACKGROUND--------------------------------
    image_path = left_image_output_path
    background_upload = process_input(image_path)
    # Convert background data to numpy array
    background_upload_np = np.array(background_upload)
    print(background_upload_np.shape)


    ###Gender
    explainer_layer2 = shap.GradientExplainer(model_layer, background_train_np)

    shap_values_left_opg_2 = explainer_layer2.shap_values(background_upload_np)
    

    shap_img = shap.image_plot(shap_values_left_opg_2, background_upload_np, show=False)
    #plt.figure(shap_img)
    print('shap done')
    #####------------------------

    # Save the SHAP image plot
    image_filename = 'shap_image_plot.png'  # Include the file extension
    image_path = os.path.join(app['config']['FOLDER_SHAP2ND'], image_filename)

    # Save the current figure
    plt.savefig(image_path)
    plt.close()  
    
# Open and display the original image
    img = Image.open(image_path)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img)
    # plt.axis('off')  # Turn off axis numbers and ticks
    # plt.show()

# Crop the image
    width, height = img.size
    frac = 0.7
    crop_up_height = int(height * frac)
    top = (height - crop_up_height) // 2
    cropped_img = img.crop((0, top, width, crop_up_height))

# Save the cropped image
    cropped_filename = 'cropped_shap_image_plot.png'
    cropped_image_path = os.path.join(app['config']['FOLDER_SHAP2ND'], cropped_filename)
    cropped_img.save(cropped_image_path)
    
    # plt.savefig(image_path)
    # plt.close()  
    
    print(f'SHAP image saved to {image_path}')

    text_ = 'DONE'
    
    return text_


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python shap_.py <left_image_output_path> <model_select> <background_train_np_path> ")
        sys.exit(1)
        
    left_image_output_path = sys.argv[1]
    model_select = sys.argv[2]
    background_train_np_path = sys.argv[3]
    
    # Load the NumPy array from the file
    background_train_np = np.load(background_train_np_path)
    
    print(f'left_image_output_path: {left_image_output_path}')
    print(f'model_select: {model_select}')
    print(f'background_train_np: {background_train_np}')
    
    main(left_image_output_path, model_select, background_train_np)
    
    