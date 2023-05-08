from flask import Flask, request, jsonify
from src.ai_model import predict

"""
Function:
        Configures routes on the provided Flask app.
Args: 
        App (Flask object): the Flask app to configure the routes on

Returns: 
        None (routes are being configured for the provided Flask app)
"""
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
