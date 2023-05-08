from flask import Flask, request, jsonify
from ai_model import predict

app = Flask(__name__)


@app.route('/process_data', methods=['POST'])
def process_data():
    # Get the JSON data from the request
    data = request.get_json()

    # Get the values of the county, age, diagnosis, and gender keys
    region = data['region']
    age = data['age']
    disorder = data['disorder']
    gender = data['gender']

    # Process the data and generate a result
    result = {'weeks': str(predict([region, age, disorder, gender]))}

    # Return the result as JSON
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)