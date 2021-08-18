import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as img
import dataframe_image as dfi
import pandas as pd
import dataframe_image as dfi
from wms import RLWMS

app = Flask(__name__)
wms = joblib.load('rlwms.pkl')
#Home Page / Initial Load
@app.route('/')
def home():
    return render_template('index.html')
#Prediction page or post method files when user clicks submit button which predicts the model output.
@app.route('/predict',methods=['POST'])
def predict():
    inValue = request.form['picklocation']
    inValues = inValue.split(',')
    inValues1 = int(inValues[0])
    inValues2 = int(inValues[1])
    if inValues1 > 29 or inValues2 > 29:
        prediction = []
    else:            
        prediction = wms.findShortestPath(inValues1, inValues2)     
    path = prediction
    pathvalue = prediction   
    df = pd.DataFrame(wms.rewards)
    def color_specific_cell(x,path):
        color = 'background-color : green'
        df1 = pd.DataFrame('',index=x.index, columns=x.columns)
        for each in path:
            df1.loc[each[0],each[1]]= color
        return df1

    if len(prediction) > 0:
        imageplt = df.style.apply(color_specific_cell, path=prediction, axis=None)    
        dimage = dfi.export(imageplt, 'static/df_styled'+str(inValues1)+str(inValues2)+'.jpg')
        return render_template('index.html', prediction_text='Shortest path is  {}'.format(path),prediction_image='static/df_styled'+str(inValues1)+str(inValues2)+'.jpg')  
    else:
        pathvalue = "invalid"
        return render_template('index.html', prediction_text='Picking Location '+ str(inValues) +' is  {}'.format(pathvalue))  
    
if __name__ == "__main__":    
    app.run(debug=True)    
    
