from flask import Flask
from . import ai_model

app = Flask(__name__)

@app.route('/')
def home():
    return ai_model.getResult(['Agder', '16-19', 'pregnancy disorders', 'woman'])

@app.route('/about')
def about():
    return 'About'