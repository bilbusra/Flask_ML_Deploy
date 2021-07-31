from flask import Flask, redirect, url_for, flash, jsonify, request
import pickle, pandas as pd
from feature_extraction import convert_data
import sys
import dill

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    jsonfile = request.get_json()
    print(jsonfile)
    data = pd.DataFrame.from_dict(jsonfile)
    print("json okundu")
    print(data)

    converted_data = convert_data(data, feature_cols)
    if 'index' not in feature_cols:
        feature_cols.append('index')
    converted_data.columns = feature_cols
    pred_results = model.predict(converted_data.drop(converted_data.columns[-1], axis=1))

    converted_data['target__office'] = pred_results

    return jsonify(pred_results.tolist())

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "tools":
            renamed_module = "whyteboard.tools"

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

if __name__ == '__main__':
    sys.path.append(r'C:\Users\Busra\.conda\pkgs\scikit-learn-0.21.2-py36h0ff8352_0\Lib\site-packages\sklearn\ensemble')
    model = dill.load(open(r'C:\Users\Busra\Desktop\data\data\model_dill.pkl', 'rb'))
    feature_cols = pickle.load(open(r'C:\Users\Busra\Desktop\data\data\features.pkl', 'rb'))
    print("loaded")
    app.run()
