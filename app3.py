import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = secrets.token_hex(16)

UPLOAD_FOLDER_IMAGES = 'static/uploads/images/'
UPLOAD_FOLDER_MODELS = 'static/uploads/models/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'h5', 'hdf5'}

app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['UPLOAD_FOLDER_MODELS'] = UPLOAD_FOLDER_MODELS

# Create upload folders if they don't exist
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MODELS, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_models():
    return [f for f in os.listdir(UPLOAD_FOLDER_MODELS) if f.endswith('.h5')]

@app.route('/', methods=['GET', 'POST'])
def index():
    model_names = get_available_models()
    return render_template('index.html', model_names=model_names)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    model_file = request.files['model_file2']
    if model_file and allowed_file(model_file.filename):
        filename = secure_filename(model_file.filename)
        model_file.save(os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename))
        flash("Model uploaded successfully", "success")
    else:
        flash("Invalid model file type", "error")
    return redirect(url_for('index'))

@app.route('/upload_image', methods=['POST'])
def predict():
    selected_model = request.form.get('selected_model')
    image_file = request.files.get('image')

    if not image_file or not allowed_file(image_file.filename):
        flash("Invalid image file type", "error")
        return redirect(url_for('index'))
    
    try:
        # Save the uploaded image file
        image_filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
        image_file.save(image_path)
        
        if not selected_model:
            flash("No model selected, image saved", "error")
            return redirect(url_for('index'))

        # Load the model
        model_path = os.path.join(app.config['UPLOAD_FOLDER_MODELS'], selected_model)
        model = load_model(model_path)

        # Preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Get predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        output = f"Prediction from {selected_model}: {predicted_class}"

        return render_template('index.html', output=output, model_names=get_available_models(), image_path=image_path)
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
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
