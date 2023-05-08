from flask import Flask
from routes import configure_routes

# Create Flask instance
app = Flask(__name__)

# Configure the routes (which is currently only /process_data)
configure_routes(app)

# Run Flask app with debug mode on 0.0.0.0
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)