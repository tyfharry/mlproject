from simple_linear_regr_utils import generate_data, evaluate
import simple_linear_regr
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def generate_prediction(X):
    """
    :param X: the input record. It can be one record or a batch of records
    :return:
    predicted: the predicted output
    """
    diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = generate_data()
    model = simple_linear_regr.SimpleLinearRegression()
    model.fit(diabetes_X_train,diabetes_y_train)
    y_hat = model.predict(X)
    return y_hat

@app.route('/stream', methods=['POST'])
def stream():
    data = request.get_json(force=True)  # force=True, ensures that a JSON is returned
    predicted = generate_prediction(data)
    return jsonify(predicted=predicted)

@app.route('/batch', methods=['POST'])
def batch():
    data = request.get_json(force=True)
    results = []
    for record in data:
        predicted = generate_prediction(record)
        results.append(predicted)
    return jsonify(results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)