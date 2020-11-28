from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import _pickle as cPickle
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)

#Array used for calibrating the house prices
calibrationArr = np.array([[5.00000000e+00,6.79905586e+00, 1.00000000e+00, 0.00000000e+00,
  1.96100000e+03],
 [6.00000000e+00, 7.19293422e+00, 1.00000000e+00, 0.00000000e+00,
  1.95800000e+03],
 [5.00000000e+00, 7.39633529e+00 ,2.00000000e+00 ,1.00000000e+00,
  1.99700000e+03],
 [6.00000000e+00, 7.38087904e+00, 2.00000000e+00, 1.00000000e+00,
  1.99800000e+03],
 [8.00000000e+00, 7.15539630e+00 ,2.00000000e+00 ,0.00000000e+00,
  1.99200000e+03]])



with open('model.json', 'rb') as f:
    forest = cPickle.load(f)
    
with open('model(1).rf', 'rb') as f:
    forestDaan = cPickle.load(f)

daanModel = load_model('model.h5')
ianModel = load_model('mushroom_predictor_new.h5', compile=False)
tomModel = load_model('tom_model.h5')

app = Flask(__name__)

@app.route('/ian/', methods=['POST'])
@cross_origin(origin='*')
def ian_ding():

    vals = request.json.get("input")
    
    pred = ianModel.predict([vals])

    return str(pred), 201
    
@app.route('/tom/', methods=['POST'])
def tom_ding():
    
    vals = request.json.get("input")
    zz = np.array([vals])
    callB = np.vstack([calibrationArr,zz])
    scale = StandardScaler()
    X_test = scale.fit_transform(callB)
    pred = tomModel.predict([X_test])
    val = int(pred[-1][0])
    #print(pred[-1][0])

    return str(val),201



@app.route('/daan/', methods=['POST'])
def daan_ding():

    vals = request.json.get("input")
    
    pred = daanModel.predict([vals])

    return str(pred[-1]), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)