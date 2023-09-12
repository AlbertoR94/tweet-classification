import requests
import json
import numpy as np

import model

url =  "http://localhost:9696/predict"

input_data = {
            'keyword':{"0":None, "1":"ablaze"},
            'text':{"0":"God, Just happened a terrible car crash!", "1":"SETTING MYSELF ABLAZE http://t.co/6vMe7P5XhC"}
            }

# Convert to json

input_data = json.dumps(input_data)

headers = {"Content-Type": "application/json"}

resp = requests.post(url, input_data, headers=headers)
print(resp.text)

#input_data = model.read_input_2(input_data)
#clf = model.load_model()
#pipe = model.load_pipeline()

#test = model.process_input(input_data)
#X = model.format_input(test)
#print(X)
#X = pipe.transform(X)
#y_pred = clf.predict_proba(X)
#pred = model.pred print(input_data['keyword'])ict_label(input_data)[0][1]
#print(X)