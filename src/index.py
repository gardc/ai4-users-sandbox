from flask import Flask, request, jsonify
from ai_model import predict
from routes import configure_routes

app = Flask(__name__)

configure_routes(app)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)