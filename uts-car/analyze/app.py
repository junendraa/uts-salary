from flask import Flask
from routes import bp as analyze_bp

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(analyze_bp)  # blueprint from routes.py
    return app

@app.route("/") 
def home(): 
    return render_template("analyze.html")

if __name__ == "__main__":
    app = create_app()
    print("🔬 Starting Analyze app on port 5002")
    app.run(debug=True, port=5002)
