from flask import Flask
from app.routes.main import bp as main_bp

app = Flask(__name__)
app.register_blueprint(main_bp)
