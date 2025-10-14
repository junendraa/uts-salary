import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_regression(X, y, model, feature_names):
    """
    Membuat visualisasi regresi linear (fit line vs data asli)
    Menyimpan hasil ke /static/plots/regression_plot.png
    """
    os.makedirs("static/plots", exist_ok=True)

    # Ambil satu fitur dominan (misalnya years_exp)
    # Misalnya fitur ke-3 (experience)
    idx_exp = feature_names.index("years_exp") if "years_exp" in feature_names else 0
    x = X[:, idx_exp]
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.7, label="Data Asli")
    plt.plot(x, y_pred, color="red", label="Garis Regresi", linewidth=2)
    plt.title("Linear Regression: Pengaruh Pengalaman terhadap Gaji")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = "static/plots/regression_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path
