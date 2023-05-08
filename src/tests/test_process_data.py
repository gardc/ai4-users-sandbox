import json

from flask import Flask

from src.routes import configure_routes


def test_index():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    # Define the request data
    data = {'region': 'iNnLaNdEt', 'age': "20-24",
            'disorder': 'mental disorderS', 'gender': 'female'}
    headers = {'Content-Type': 'application/json'}

    # Send a POST request to the endpoint with the request data
    response = client.post(
        '/process_data', data=json.dumps(data), headers=headers)

    # Assert the response status code and content
    assert response.status_code == 200
    assert response.content_type == 'application/json'
    assert json.loads(response.data) == {'weeks': '6'}

    # Test with different inputs
    data = {'region': 'NORDLAND', 'age': "20-24",
            'disorder': 'mental disorderS', 'gender': 'female'}
    response = client.post(
        '/process_data', data=json.dumps(data), headers=headers)
    assert response.status_code == 200
    assert response.content_type == 'application/json'
    assert json.loads(response.data) == {'weeks': '7'}
