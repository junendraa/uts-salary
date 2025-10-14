import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import Flask, render_template, request

# =====================================================
# 🚀 BAGIAN 1 — PIPELINE TRAINING MODEL (DATASET GAJI)
# =====================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_and_preprocess_data(path):
    """Load & preprocess salary dataset"""
    print_header("STEP 1: LOAD & PREPROCESS DATA")

    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📊 Columns found: {list(df.columns)}")

    # Ubah sesuai dataset kamu
    if "salary" not in df.columns:
        raise KeyError("Kolom 'salary' tidak ditemukan di dataset.")

    X = df.drop("salary", axis=1)
    y = df["salary"]

    return X, y


def split_and_scale_data(X, y):
    """Split data"""
    print_header("STEP 2: TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train model"""
    print_header("STEP 3: TRAINING MODEL")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model"""
    print_header("STEP 4: MODEL EVALUATION")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    return {
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }


def visualize_results(y_train, y_test, y_train_pred, y_test_pred):
    """Visualize results"""
    print_header("STEP 5: VISUALIZATION")

    os.makedirs("static/images", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_test_pred, alpha=0.7, edgecolors="k")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Test Set: Actual vs Predicted")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Perfect Fit",
    )
    plt.legend()
    plt.tight_layout()
    path = "static/images/model_evaluation.png"
    plt.savefig(path)
    plt.close()
    print(f"✅ Visualization saved at {path}")


def save_artifacts(model):
    """Save model"""
    print_header("STEP 6: SAVING MODEL")
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model saved to models/model.pkl")


def run_training_pipeline():
    """Main pipeline execution"""
    DATA_PATH = "data/raw/data.csv"
    if not os.path.exists(DATA_PATH):
        print(f"❌ File not found: {DATA_PATH}")
        sys.exit(1)

    X, y = load_and_preprocess_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    visualize_results(y_train, y_test, results["y_train_pred"], results["y_test_pred"])
    save_artifacts(model)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎯 Open browser at http://127.0.0.1:5000 to test Flask UI.")


# =====================================================
# 🚀 BAGIAN 2 — FLASK WEB APP (FITUR 1 + FITUR 2)
# =====================================================

app = Flask(__name__)
UPLOAD_FOLDER = "data/uploads"
PLOT_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    """Fitur 1 — Prediksi salary (user input individu)"""
    predicted_salary = None
    form_data = {}

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            gender = request.form["gender"]
            education = request.form["education"]
            experience = float(request.form["experience"])

            gender_val = 1 if gender.lower() == "male" else 0
            edu_map = {
                "High School": 0,
                "Bachelor's Degree": 1,
                "Master's Degree": 2,
                "PhD": 3,
            }
            edu_val = edu_map.get(education, 0)

            # Dummy formula sementara
            predicted_salary = round(3000 + (experience * 1200) + (edu_val * 800), 2)

            form_data = {
                "age": int(age),
                "gender": gender,
                "education": education,
                "experience": int(experience),
            }
        except Exception as e:
            predicted_salary = f"Error: {str(e)}"

    return render_template("index.html", predicted_salary=predicted_salary, form_data=form_data)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """Fitur 2 — Upload CSV dan tampilkan hasil regresi"""
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


# =====================================================
# 🚀 ENTRY POINT — PILIH MODE
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run Flask web app instead of training pipeline"
    )
    args = parser.parse_args()

    if args.web:
        print("🌐 Starting Flask web app on port 5001...")
        app.run(debug=True, port=5001)
    else:
        run_training_pipeline()
