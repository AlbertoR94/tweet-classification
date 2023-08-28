from flask import Flask, request, jsonify
import model
import pandas as pd

app = Flask(__name__)

clf = model.load_model()
pipe = model.load_pipeline()
train = pd.read_csv('processed_train.csv')

@app.route('/')
def hello():
    html = f"Predict whether a tweet is fake or real"
    return html.format(format)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = model.read_input(request.json)
    pred = model.predict_label(input_data)
    json = jsonify({'0':list(pred[:, 0]), '1':list(pred[:,1])})
    print(json.json)
    return "Done" 