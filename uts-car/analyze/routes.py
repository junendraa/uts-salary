import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, redirect, url_for
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

bp = Blueprint("analyze", __name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
PLOT_FOLDER = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

@bp.route("/", methods=["GET", "POST"])
def index():
    error = None
    regression_info = None
    plot_relpath = None

    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or f.filename == "":
            error = "Please upload a CSV file."
        else:
            # save uploaded file
            save_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(save_path)

            try:
                df = pd.read_csv(save_path)
                # minimal required columns: years_exp, salary
                if "years_exp" not in df.columns or "salary" not in df.columns:
                    error = "CSV harus memiliki kolom 'years_exp' dan 'salary'."
                else:
                    X = df[["years_exp"]].values.reshape(-1, 1)
                    y = df["salary"].values

                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)

                    # save plot
                    plt.figure(figsize=(8,5))
                    plt.scatter(X, y, alpha=0.7, label="Data asli")
                    # sort by X for line plotting
                    idx = X[:,0].argsort()
                    plt.plot(X[idx,0], y_pred[idx], color="red", linewidth=2, label="Garis regresi")
                    plt.xlabel("Years of Experience")
                    plt.ylabel("Salary")
                    plt.title("Linear Regression on Uploaded Dataset")
                    plt.legend()
                    plt.tight_layout()

                    plot_filename = f"regression_{os.path.splitext(f.filename)[0]}.png"
                    plot_path = os.path.join(PLOT_FOLDER, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()

                    r2 = r2_score(y, y_pred)
                    regression_info = {
                        "coef": float(model.coef_[0]),
                        "intercept": float(model.intercept_),
                        "r2": float(r2),
                        "n_samples": len(df)
                    }
                    # web relative path for template
                    plot_relpath = url_for("analyze.static", filename=f"plots/{plot_filename}")

            except Exception as e:
                error = f"Error processing file: {e}"

    return render_template("analyze.html",
                           error=error,
                           regression_info=regression_info,
                           plot_path=plot_relpath)
