import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

#path img
img_path = 
# function to predict age
def predict_age(my_models[0], img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    predictions1 = my_models[0].predict(x)
   # Convert predictions to class labels
    if predictions1[0][0]==0:
        Age = 'Younger'
    else:
        Age = 'Older'

    return Age

# function to predict gender
def predict_gender(my_models[1], img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    predictions2 = my_models[1].predict(x)

    # Convert predictions to class labels
    if predictions2[0][0] == 0:
        gender = 'Female'
    else:
        gender = 'Male'

    return gender



# predictions
if selected_model == '0':
  predicted_model = predict_age(my_models[0], img_path)
else:  
  predicted_model = predict_gender(my_models[1], img_path)

# Print predictions
print('Predicted age:', predicted_age)
print('Predicted gender:', predicted_gender)




# predictions
predicted_age = predict_age(model_layer1, img_path)
predicted_gender = predict_gender(model_layer2, img_path)

# Print predictions age
print('Predicted age:', predicted_age)
# Print predictions gender
print('Predicted gender:', predicted_gender)


