from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MODEL_FOLDER'] = 'models/'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['ALLOWED_MODEL_EXTENSIONS'] = {'h5', 'hdf5', 'pb', 'pbtxt', 'tflite'}

# Ensure the upload and model folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Load your models
models = {}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((128, 128))  # Adjust size as needed
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file2' not in request.files:
        return redirect(request.url)
    file = request.files['model_file2']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename, app.config['ALLOWED_MODEL_EXTENSIONS']):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
        file.save(filepath)
        model_name = filename.rsplit('.', 1)[0]
        models[model_name] = load_model(filepath)
        return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        selected_model = request.form.get('selected_model')
        if selected_model not in models:
            return "Model not found", 400
        
        image = preprocess_image(filepath)
        model = models[selected_model]
        prediction = model.predict(image)  # Adjust based on your model's output
        
        # Assuming the model returns a probability for a binary classification
        prediction_result = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        return render_template('predict.html', prediction=prediction_result, filename=filename)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
