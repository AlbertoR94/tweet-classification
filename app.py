from flask import Flask, request, jsonify
from utils import generate_dictionary
import model
import pandas as pd

app = Flask(__name__)

clf = model.load_model()
pipe = model.load_pipeline()
train = pd.read_csv('processed_train.csv')

@app.route('/')
def hello():
    html = "Predict whether a tweet is fake or real"
    return html.format(format)

@app.route('/predict', methods=['POST'])
def predict():
    #print(type(request.json))
    input_data = model.read_input(request.json)
    preds = model.predict_label(input_data)
    results = generate_dictionary(preds)
    #print(results)
    json = jsonify(results)
    return json

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)