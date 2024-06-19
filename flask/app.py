import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import sys
from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects
import numpy as np   
from werkzeug.utils import secure_filename
import keras
print(keras.__version__)
from PIL import Image
print(Image.__version__)
import cv2
import subprocess
import torch
from yolov5 import utils



app = Flask(__name__, static_folder='static')  # Include static folder
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Configuration (adjust as needed)
FOLDER_MY_MODELS = 'static/my_models'
UPLOAD_FOLDER_IMAGES = 'static/uploads/images'
UPLOAD_FOLDER_MODELS = 'static/uploads/models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'h5'}

app.config['FOLDER_MY_MODELS'] = FOLDER_MY_MODELS
app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['UPLOAD_FOLDER_MODELS'] = UPLOAD_FOLDER_MODELS

# Create upload folders if they don't exist
os.makedirs(FOLDER_MY_MODELS, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MODELS, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_models():
    return [f for f in os.listdir(FOLDER_MY_MODELS) if f.endswith('.h5')]

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

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #model
    model_file = request.files.get('model_file2')
    filename = None
    input_model = None
    
    if model_file and allowed_file(model_file.filename):
        filename = model_file.filename
        model_file.save(os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename))
        input_model = os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename)
    
    model_select_input = request.form.get('model_select')
    predict_input_page1 = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')
    
    my_models = get_available_models()
    if model_select_input == '0':  # Age estimation model
        model = os.path.join(app.config['FOLDER_MY_MODELS'], my_models[0]) if my_models else None
    elif model_select_input == '1':  # Sex estimation model
        model = os.path.join(app.config['FOLDER_MY_MODELS'], my_models[1]) if len(my_models) > 1 else None
    else:
        model = None
    
    selected_model = input_model if input_model else model
    print(f'selected_model: {selected_model}')
    image_file = request.files.get('image')
    
    if not selected_model:
        #flash("No model selected, image saved", "error")
        return redirect(url_for('index'))

    if selected_model and image_file and allowed_file(image_file.filename):
        # Save the uploaded image file
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
        image_file.save(image_path)
       
        # Define the output filenames
        left_image_filename = 'left_' + image_filename
        left_image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], left_image_filename)
        print(left_image_path)

        # call the cut_image.py script as a subprocess
        subprocess.run(['python', 'cut_image.py', image_path, left_image_path])
        
        # Load the model
        model_path = selected_model
        
        print(f"Selected model: {selected_model}")
        print(f"Image filename: {image_filename}")
        print(f"Image path: {image_path}")
        print(f"Model path: {model_path}")
        
        try:
            model = load_custom_model(model_path)
            model.summary()
            # You can proceed with predictions here
        except Exception as e:
            print(f"Error loading model: {e}")
            return jsonify({'error': str(e)}), 500

        #Preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size as per your model's requirement
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image if required by your model

        # Get predictions
        predictions = model.predict(img_array)
        # Assuming the model's output is a single class prediction, modify as needed
        predicted_class = np.argmax(predictions[0])
        output = f"Prediction from {selected_model}: {predicted_class}"
        print(f'output: {output}')

        # call the shap.py script as a subprocess
        subprocess.run(['python', 'shap.py', left_image_path])
        
        return render_template('shappage.html', output=output, predict_input=predict_input_page1, node0_input=node0_input, node1_input=node1_input)
    
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
    
