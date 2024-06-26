import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import sys
# from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects
import numpy as np   


import keras
print(keras.__version__)

app = Flask(__name__, static_folder='static')  # Include static folder
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Configuration (adjust as needed)
UPLOAD_FOLDER_IMAGES = 'static/uploads/images'
UPLOAD_FOLDER_MODELS = 'static/uploads/models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'h5'}

app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['UPLOAD_FOLDER_MODELS'] = UPLOAD_FOLDER_MODELS

# Create upload folders if they don't exist
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MODELS, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_models():
    return [f for f in os.listdir(UPLOAD_FOLDER_MODELS) if f.endswith('.h5')]

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
    model_names = get_available_models()
    return render_template('index.html', model_names=model_names)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    model_file = request.files['model_file2']
    if model_file and allowed_file(model_file.filename):
        filename = model_file.filename
        model_file.save(os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename))
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form['selected_model']
    image_file = request.files['image']

    if selected_model and image_file and allowed_file(image_file.filename):
        # Save the uploaded image file
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
        image_file.save(image_path)
        
        # Load the model
        model_path = os.path.join(app.config['UPLOAD_FOLDER_MODELS'], selected_model)
        
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
        
        # Preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size as per your model's requirement
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image if required by your model

        # Get predictions
        predictions = model.predict(img_array)
        # Assuming the model's output is a single class prediction, modify as needed
        predicted_class = np.argmax(predictions[0])
        output = f"Prediction from {selected_model}: {predicted_class}"

        return render_template('index.html', output=output, model_names=get_available_models())
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)