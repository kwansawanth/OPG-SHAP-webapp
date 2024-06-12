from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Initializer
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import logging

class ConvKernalInitializer(Initializer):
    def __call__(self, shape, dtype=None):
        return tf.random.normal(shape, dtype=dtype)  # Or another standard initialization


# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
app.config['UPLOAD_FOLDER'] = 'static/uploads/images/'
app.config['MODEL_FOLDER'] = 'static/uploads/models/'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['ALLOWED_MODEL_EXTENSIONS'] = {'h5', 'hdf5'}

# Ensure the upload and model folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Load your models
models = {}
default_models = {
    'age_model': '16_Multi_8e-6_250_Unfreeze(age).h5',
    'gender_model': '11_Multi_2e-4_250_Unfreeze_Gender.h5'
}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((128, 128))  # Adjust size as needed
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

class ConvKernalInitializer(Initializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape, dtype=None):
        fan_out = shape[-1]
        fan_in = shape[-2] * shape[0] * shape[1]
        scale = 1.0 / max(1., (fan_in + fan_out) / 2.0)
        limit = tf.sqrt(3.0 * scale)
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

    def get_config(self):
        return {}

# Load all models from the models folder
def load_all_models():
    for filename in os.listdir(app.config['MODEL_FOLDER']):
        if allowed_file(filename, app.config['ALLOWED_MODEL_EXTENSIONS']):
            model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
            try:
                model_name = filename.rsplit('.', 1)[0]
                models[model_name] = load_model(model_path, custom_objects={'ConvKernalInitializer': ConvKernalInitializer})
                logging.info(f'Models {model_name} loaded successfully')
            except Exception as e:
                logging.error(f'Error loading model {filename}: {e}')

load_all_models()

@app.route('/')
def index():
    return render_template('index.html', models=models.keys())

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file2' not in request.files:
        flash('No file part')
        logging.error('No file part in the request')
        return redirect(url_for('index'))
    file = request.files['model_file2']
    if file.filename == '':
        flash('No selected file')
        logging.error('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename, app.config['ALLOWED_MODEL_EXTENSIONS']):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
        try:
            file.save(filepath)
            model_name = filename.rsplit('.', 1)[0]
            models[model_name] = load_model(filepath, custom_objects={'ConvKernalInitializer': ConvKernalInitializer})
            flash('Model successfully uploaded and loaded')
            logging.info(f'Model {model_name} loaded successfully')
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")
            flash(f"An error occurred while loading the model: {e}")
            return redirect(url_for('index'))
        return redirect(url_for('index'))
    else:
        flash('File type not allowed')
        logging.error('File type not allowed')
        return redirect(url_for('index'))

@app.route('/upload_image', methods=['POST'])
def up_image():
    if 'image' not in request.files:
        flash('No file part')
        logging.error('No file part in the request')
        print("1")
        return redirect(url_for('shappage')) #มีรูปแล้ว
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        logging.error('No selected file')
        print("2")
        return redirect(url_for('index'))
    if file and allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        selected_model = request.form.get('selected_model')
        logging.info(f'Selected model: {selected_model}')
        logging.info(f'Available models: {list(models.keys())}')

        if selected_model not in models:
            flash('Model not found')
            logging.error(f'Model {selected_model} not found')
            print("3")
            return redirect(url_for('index'))
        
        model = models[selected_model]

        image = preprocess_image(filepath)
        try:
            prediction = model.predict(image)  # Adjust based on your model's output
            
            # Assuming the model returns a probability for a binary classification
            prediction_result = "Positive" if prediction[0][0] > 0.5 else "Negative"
            logging.info(f'Prediction result: {prediction_result}')
            return render_template('predict.html', prediction=prediction_result, filename=filename)
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            flash(f"An error occurred during prediction: {e}")
            return redirect(url_for('index'))
    else:
        flash('File type not allowed')
        logging.error('File type not allowed')
        return redirect(url_for('index'))

@app.route('/shappage')
def shappage():
    return render_template('shappage.html')

@app.route('/detectionpage')
def detectionpage():
    return render_template('detectionpage.html')

@app.route('/evaluationpage')
def evaluationpage():
    return render_template('evaluationpage.html')

if __name__ == '__main__':
    app.run(debug=True)
