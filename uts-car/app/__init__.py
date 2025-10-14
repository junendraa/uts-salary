from flask import Flask
from app import routes

def create_app():
    app = Flask(__name__)

    # Import routes
    from app import routes
    app.register_blueprint(routes.bp)

    return app

