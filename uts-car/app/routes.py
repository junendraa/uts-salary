from flask import Blueprint, render_template, request
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

bp = Blueprint("routes", __name__)

UPLOAD_FOLDER = "data/uploads"
PLOT_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

@bp.route("/", methods=["GET", "POST"])
def index():
    # halaman prediksi salary kamu yang lama
    return render_template("index.html")

@bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    regression_info = None
    plot_path = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please upload a CSV file."
        else:
            try:
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
                uploaded_file.save(file_path)
                df = pd.read_csv(file_path)

                if "years_exp" not in df.columns or "salary" not in df.columns:
                    error = "CSV must have 'years_exp' and 'salary' columns."
                else:
                    X = df[["years_exp"]]
                    y = df["salary"]

                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)

                    plt.figure(figsize=(6, 4))
                    plt.scatter(X, y, color="blue", label="Data Asli")
                    plt.plot(X, y_pred, color="red", linewidth=2, label="Regresi Linear")
                    plt.xlabel("Years of Experience")
                    plt.ylabel("Salary")
                    plt.title("Linear Regression on Uploaded Dataset")
                    plt.legend()
                    plt.tight_layout()

                    plot_filename = "regression_plot.png"
                    plot_path = os.path.join(PLOT_FOLDER, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()

                    r2 = r2_score(y, y_pred)
                    regression_info = {
                        "coef": float(model.coef_[0]),
                        "intercept": float(model.intercept_),
                        "r2": float(r2),
                        "n_samples": len(df),
                    }

            except Exception as e:
                error = f"Error processing file: {str(e)}"

    return render_template(
        "analyze.html",
        regression_info=regression_info,
        plot_path=plot_path,
        error=error,
    )
