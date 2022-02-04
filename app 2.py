import glob
import numpy as np
import os

# Importing deep-learning related libraries:
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Importing flask-related libraries:
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Initializing our flask application:
app = Flask(__name__)

modelPath = r'C:\Users\sahas\OneDrive\Documents\visual studio code\medicare contrib\covid_detect.h5'

# Loading our model:
model = load_model(modelPath)

def modelPredict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Preprocessing the image:
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
        
    predictions = model.predict(x)
    predictions = np.argmax(predictions, axis=1)
    if predictions == 0:
        predictions = "Normal"
    elif predictions == 1:
        predictions = "Covid"
        
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Saving the file to an "upload folder":
        basePath = os.path.dirname(__file__)
        filePath = os.path.join(
            basePath, 'uploads', secure_filename(f.filename)
        )
        f.save(filePath)
        
        predictions = modelPredict(filePath, model)
        result = predictions
        return result
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)