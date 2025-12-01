import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import boxcox, gaussian_kde

# === Einstellungen ===
plan = 'Taguchi' # Relevant nur für Pools
submodel_nummer = '' # Für Model 1: 1 (-4/-3(daten)), 2 (-3/-2(daten)) oder 3 (-2/-1(daten)). Für Model 2: ""
plan_nummer = '2'      # Model 1 oder Model 2

# Auskommentieren je nachdem ob model, pool oder holdout ausgewertet wird
#excel_path_pools = f"Getrennte_Daten/Pool_{plan}_Modell_{plan_nummer}.xlsx"
excel_path_holdout = f"Getrennte_Daten/Holdout_fixed_Modell_{plan_nummer}.xlsx"
#excel_path_data_model_1 = f"Daten/{plan}_Einleger_Fließfronten.xlsx"
#excel_path_data_model_2 = f"Daten/{plan}_Fließfronten_Auslenkung.xlsx"

# === Ausgabeordner ===
#plot_ordner = f"Plots_Distribution/{plan}" # Versuchsplan
plot_ordner = f"Plots_Distribution/Holdout" # Pool oder Holdout
os.makedirs(plot_ordner, exist_ok=True)

# === Datei laden ===
df = pd.read_excel(excel_path_holdout) # excel_path anpassen
dateiname = os.path.splitext(os.path.basename(excel_path_holdout))[0] # excel_path anpassen

# === Zu analysierende Spalte (anpassen: -4, -3, -2)
data = df.iloc[:, -2].astype(float)  # <- -4 (Submodell 1), -3 (Submodell 2), -2 (Submodell 3), -2 (Modell 2) für Pool/Holdout
#data = df.iloc[:, -1].astype(float)  # <- -3 (Submodell 1), -2 (Submodell 2), -1 (Submodell 3), -1 (Modell 2) für Daten

def plot_histogram(data, title_suffix, filename_suffix):
    """Erstellt und speichert ein Histogramm mit KDE."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(data, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black", label="Histogramm")

    try:
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 500)
        ax.plot(x_vals, kde(x_vals), color="black", linewidth=2, label="KDE")
    except Exception:
        pass

    ax.set_title(f"Histogramm – {dateiname}_{plan_nummer}_{submodel_nummer}{title_suffix}")
    ax.set_xlabel("Wert")
    ax.set_ylabel("Dichte")
    ax.legend()
    # Auskommentieren je nachdem ob daten, pool oder holdout ausgewertet wird
    #fig.savefig(os.path.join(plot_ordner, f"Histogramm_Pool_{plan}_Modell_{plan_nummer}_{submodel_nummer}_{filename_suffix}.png"))
    fig.savefig(os.path.join(plot_ordner, f"Histogramm_Holdout_fixed_Modell_{plan_nummer}_{submodel_nummer}_{filename_suffix}.png"))
    #fig.savefig(os.path.join(plot_ordner, f"Histogramm_{plan}_Modell_{plan_nummer}_{submodel_nummer}_{filename_suffix}.png"))

    plt.close(fig)

# === Originaldaten ===
plot_histogram(data, "", "original")

# === Log-Transformation ===
data_log = np.log1p(data - np.min(data) + 1e-6) if np.any(data <= 0) else np.log1p(data)
plot_histogram(data_log, " (Log)", "log")

# === Box-Cox-Transformation ===
data_bc = data - np.min(data) + 1e-6 if np.any(data <= 0) else data
data_bc, lambda_bc = boxcox(data_bc)
plot_histogram(data_bc, f" (Box-Cox λ={lambda_bc:.3f})", "boxcox")

print("Histogram wurden erfolgreich erstellt.")