from logging import debug
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Loading our diabetes model:
modelDB = pickle.load(open("models/LogModelDiabetes.pkl", "rb"))
modelBC = pickle.load(open("models/RFModelCancer.pkl", "rb"))

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
            
    return render_template('diabetes-result.html', prediction_text = result)

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
            
    return render_template('cancer-result.html', prediction_text = result)

if __name__ == '__main__':
    app.run(debug=True)