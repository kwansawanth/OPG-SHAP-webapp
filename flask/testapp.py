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


app = Flask(_name_, static_folder='static')  # Include static folder
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
    
    # Save the uploaded image file
    image_filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
    image_file.save(image_path)
    
    if not selected_model:
        #flash("No model selected, image saved", "error")
        return redirect(url_for('index'))

    if selected_model and image_file and allowed_file(image_file.filename):
        # Save the uploaded image file
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
        image_file.save(image_path)
        
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
        # Open and crop the image
        img = Image.open(image_path)
        width, height = img.size
        frac = 0.6
        # Crop 60% from the left of the image
        crop_left_width = int(width * frac)
        cropped_left = img.crop((0, 0, crop_left_width, height))
        cropped_left.save(left_image_output_path)
        # Preprocess the image
        img = image.load_img(left_image_output_path, target_size=(224, 224))  # Adjust target size as per your model's requirement
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image if required by your model

print("Preprocessed image shape:", img_array.shape)

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
        
    return render_template('shappage.html', predict_input=predict_input_page1, node0_input=node0_input, node1_input=node1_input)

# @app.route('/upload_model', methods=['POST'])
# def upload_model():
#     model_file = request.files['model_file2']
#     if model_file and allowed_file(model_file.filename):
#         filename = model_file.filename
#         model_file.save(os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename))
#     return redirect(url_for('index'))


# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     selected_model = request.form.get('model_select')
#     image_file = request.files.get('image')
    
#     if not image_file or not allowed_file(image_file.filename):
#         #flash ("Invalid image file type", "error")
#         return redirect(url_for('index'))
    
#     try:
#         # Save the uploaded image file
#         image_filename = secure_filename(image_file.filename)
#         print('A')
#         image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
#         image_file.save(image_path)
        
#         if not selected_model:
#             #flash("No model selected, image saved", "error")
#             return redirect(url_for('index'))

#         # Load the model
#         model_path = os.path.join(app.config['UPLOAD_FOLDER_MODELS'], selected_model)
#         model = load_model(model_path)

#         # Preprocess the image
#         img = image.load_img(image_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0

#         # Get predictions
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions[0])
#         output = f"Prediction from {selected_model}: {predicted_class}"

#         return render_template('index.html', output=output, model_names=get_available_models(), image_path=image_path)
#     except Exception as e:
#         #flash(f"An error occurred: {str(e)}", "error")
#         return redirect(url_for('index'))
    
# @app.route('/frompredict-page1', methods=['POST'])
# def predict_box():
#     if request.method == 'POST':
#         global model_select_input, predict_input_page1, node0_input, node1_input
#         model_select_input = request.form.get('model_select')
#         predict_input_page1 = request.form.get('frompredict')
#         node0_input = request.form.get('node0input')
#         node1_input = request.form.get('node1input')

#         print(f'model_select_input: {model_select_input}')
#         print(f'predict_input_page1: {predict_input_page1}')
#         print(f'node0_input: {node0_input}')
#         print(f'node1_input: {node1_input}')
#         return render_template('shappage.html', predict_input=predict_input_page1, node0_input=node0_input, node1_input=node1_input)
#     else:
#         return redirect(url_for('index'))

# @app.route('/predict', methods=['POST'])
# def predict():
#     global model_select_input, predict_input_page1, node0_input, node1_input
    
#     selected_model = request.form['selected_model']
#     image_file = request.files['image']

#     model_select_input = request.form.get('model_select')
#     predict_input_page1 = request.form.get('frompredict')
#     node0_input = request.form.get('node0input')
#     node1_input = request.form.get('node1input')

#     print(f'model_select_input: {model_select_input}')
#     print(f'predict_input_page1: {predict_input_page1}')
#     print(f'node0_input: {node0_input}')
#     print(f'node1_input: {node1_input}')
        
#     if selected_model and image_file and allowed_file(image_file.filename):
#         # Save the uploaded image file
#         image_filename = image_file.filename
#         image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
#         image_file.save(image_path)
        
#         # Load the model
#         model_path = os.path.join(app.config['UPLOAD_FOLDER_MODELS'], selected_model)
        
#         print(f"Selected model: {selected_model}")
#         print(f"Image filename: {image_filename}")
#         print(f"Image path: {image_path}")
#         print(f"Model path: {model_path}")
        
#         try:
#             model = load_custom_model(model_path)
#             model.summary()
#             # You can proceed with predictions here
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return jsonify({'error': str(e)}), 500
        
#         # Preprocess the image
#         img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size as per your model's requirement
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0  # Normalize the image if required by your model

#         # Get predictions
#         predictions = model.predict(img_array)
#         # Assuming the model's output is a single class prediction, modify as needed
#         predicted_class = np.argmax(predictions[0])
#         output = f"Prediction from {selected_model}: {predicted_class}"

#         return render_template('index.html', output=output)
#     # return redirect(url_for('index'))
#     return render_template('shappage.html')


if _name_ == '_main_':
    app.run(debug=True)
เขียนถึง โยนลิ้งโปรเจค