import tensorflow as tf
print(tf.__version__)

from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your trained model
model_path = "https://drive.usercontent.google.com/download?id=1P5Dfwzwjt_8hRf8tLh-1FUQ6nd7IoaLm&export=download&authuser=0&confirm=t&uuid=415c0448-5794-47a2-a269-51b13a080ae3&at=APZUnTUugbEs6xIFwS9vAKrWPNst:1717990709784"
model = load_model(model_path)

# Prepare image for the model
def prepare_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Assuming your model expects images in the range [0, 1]
    return img_array

predictions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image with the model
        img_array = prepare_image(filepath)
        prediction = model.predict(img_array)
        result = np.argmax(prediction, axis=1)  # Assuming a classification model

        # Store the prediction
        predictions[filename] = int(result[0])

        return jsonify({'message': 'File successfully uploaded', 'filename': filename, 'prediction': int(result[0])})

@app.route('/output')
def output():
    return render_template('output.html', predictions=predictions)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
