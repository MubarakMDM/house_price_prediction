import pickle
from flask import Flask, request, app, jsonify, url_for,render_template
import numpy as np
import pandas as pd

# Starting point of app
app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

#got to home page
@app.route('/')
#here is the home page, once we hit his app this goes to home.html page
def home():
    return render_template('home.html')

# POST : send a request and get the output
@app.route('/predict_api', methods = ['POST'])
#here is hte definition of predict_api
def predict_api():
    #we will give 'data' as inpout in json formate and result will be stored in dataValue
    dataValue = request.json['data']
    print(dataValue)
    print(np.array(list(dataValue.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(dataValue.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data =  [float(x) for x in request.form.values()]
    print(data)
    new_data = scalar.transform(np.array(list(data)).reshape(1,-1))
    output = regmodel.predict(new_data)
    return render_template('home.html', predictionResult="The predicted value is {}".format((output[0])))  

if __name__ == '__main__' :
    app.run(debug=True)
