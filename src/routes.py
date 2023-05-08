from flask import Flask, request, jsonify
from src.ai_model import predict


def configure_routes(app):
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
