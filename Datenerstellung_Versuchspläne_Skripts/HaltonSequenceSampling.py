import numpy as np
from scipy.stats import qmc
import pandas as pd

# Anzahl der Faktoren und Stichproben
factors = 10
samples = 1000
x_scale = [-350, 350]  # X-Position
y_scale = [-150, 150]  # Y-Position
geo_scale = [0, 700]   # Geometrievektoren
seed = 42  # Seed für Reproduzierbarkeit

def create_halton_design(n_factors, n_samples):
    """Erzeugt ein Halton Sequenz Design."""
    sampler = qmc.Halton(d=n_factors, scramble=True, seed=seed)
    return sampler.random(n=n_samples)

if __name__ == "__main__":
    halton_design = create_halton_design(factors, samples)

    # Array initialisieren, um das skalierte Design zu speichern
    halton_scaled = np.zeros_like(halton_design)

    # Faktor 1 skalieren (X-Position)
    halton_scaled[:, 0] = halton_design[:, 0] * (x_scale[1] - x_scale[0]) + x_scale[0]
    # Faktor 2 skalieren (Y-Position)
    halton_scaled[:, 1] = halton_design[:, 1] * (y_scale[1] - y_scale[0]) + y_scale[0]
    # Faktoren 3 bis 10 skalieren (Geometrievektoren)
    halton_scaled[:, 2:] = halton_design[:, 2:] * (geo_scale[1] - geo_scale[0]) + geo_scale[0]

    columns = ['X_Position', 'Y_Position'] + [f'Geom{i}' for i in range(1, 9)]
    halton_df = pd.DataFrame(halton_scaled, columns=columns)

    # Den DataFrame in eine Excel-Datei speichern
    halton_df.to_excel('Halton_Design.xlsx', index=False)

    # Die Daten auch als CSV-Datei speichern
    halton_df.to_csv("Halton_Design.csv", index=False)

    # Skalierung überprüfen
    np.set_printoptions(precision=2)
    print("Erste 5 Stichproben vom skalierten Halton Design:")
    print(halton_scaled[:5, :])

    print("Halton Design wurde in 'Halton_Design.xlsx' und 'Halton_Design.csv' gespeichert.")