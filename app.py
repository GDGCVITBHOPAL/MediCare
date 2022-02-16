from logging import debug
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename 
import pickle
import numpy as np
import os
import pandas as pd

import tensorflow as tf
# Importing libraries for Keras:
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

modelPath = 'models\covid_detect.h5'

# Loading our diabetes model:
modelDB = pickle.load(open("models/LogModelDiabetes.pkl", "rb"))
modelBC = pickle.load(open("models/RFModelCancer.pkl", "rb"))
# Loading our COVID-19 model
modelCV = load_model(modelPath)

# Creating function for predicting from our model:
def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size = (224, 224))
    
    # Preprocessing the image:
    x = image.img_to_array(img)
    # Scaling our image:
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    predictions = model.predict(x)
    predictions = np.argmax(predictions, axis=1)
    
    if predictions == 1:
        predictions = "The patient doesn't have COVID."
    else:
        predictions = "The patient has COVID."
        
    return predictions

# Setting routes for our web-pages:
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/diabetes")
def diabetes_prediction():
    return render_template("diabetes.html")


@app.route("/cancer")
def cancer_prediction():
    return render_template("cancer.html")

@app.route("/covid")
def covid_prediction():
    return render_template("covid.html")


@app.route("/diabetes-predict", methods=["GET", "POST"])
def db_prediction():
    if request.method == "POST":

        pregnancies = int(request.form["pregnancies"])
        glucose = int(request.form["glucose"])
        bp = int(request.form["bp"])
        skin_thickness = int(request.form["skin_thickness"])
        insulin = int(request.form["insulin"])
        bmi = int(request.form["bmi"])
        dpf = int(request.form["dpf"])
        glucose = int(request.form["glucose"])

        predictions = modelDB.predict([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, glucose]])
        output = predictions[0]

        if output == 0:
            result = "The patient doesn't have Diabetes"
        else:
            result = "The patient has Diabetes"

    return render_template('diabetes-result.html', prediction_text=result)


@app.route("/cancer-predict", methods=["GET", "POST"])
def bc_prediction():
    if request.method == "POST":

        cpm = float(request.form["cpm"])
        area = float(request.form["area"])
        radius = float(request.form["radius"])
        perimeter = float(request.form["perimeter"])
        concavity = float(request.form["concavity"])

        predictions = modelBC.predict([[cpm, area, radius, perimeter, concavity]])
        output = predictions[0]

        if output == 0:
            result = "Type of cells: Benign. Hence the patient is cancer-free"
        else:
            result = "Type of cells: Malign. Hence the patient has cancer"

    return render_template('cancer-result.html', prediction_text=result)

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basePath = os.path.dirname(__file__)
        file_path = os.path.join(
            basePath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Making our prediction:
        prediction = model_predict(file_path, modelCV)
        result = prediction
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
