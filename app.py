from flask import Flask, request, render_template
from flask import Response
import pickle 
import numpy as np
import pandas as pd

applicaton=Flask(__name__)
app = applicaton

scaler = pickle.load(open(r'C:\Users\rohit\OneDrive\Desktop\Pregard_Pro\Model\standardScalar.pkl', "rb"))
model = pickle.load(open(r'C:\Users\rohit\OneDrive\Desktop\Pregard_Pro\Model\predectionmodel.pkl', "rb"))

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    result = ""
    if request.method == 'POST':  
    
        temperature = float(request.form.get("temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        DC = float(request.form.get("DC"))
        ISI = float(request.form.get("ISI"))
        BUI = float(request.form.get("BUI"))
        FWI = float(request.form.get("FWI"))

        
        new_data = scaler.transform([[temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI]])
        prediction = model.predict(new_data)

    
        if prediction[0] == 1:
            result = 'FIRE'
        else:
            result = 'NO FIRE'
        

        return render_template('singlepredection.html', result=result)
    else:
    
        return render_template("home.html")

if __name__ == "__main__":  
    app.run(host="0.0.0.0")
   

