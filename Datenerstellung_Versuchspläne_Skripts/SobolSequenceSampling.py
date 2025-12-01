import numpy as np
from scipy.stats import qmc
import pandas as pd

# Anzahl der Faktoren und Stichproben
factors = 10
samples = 1024
x_scale = [-350, 350]  # X-Position
y_scale = [-150, 150]  # Y-Position
geo_scale = [0, 700]    # Geometrievektoren
seed = 42 # Seed für Reproduzierbarkeit

def create_sobol_design(n_factors, n_samples):
    """Erzeugt ein Sobol Sequenz Design."""
    sampler = qmc.Sobol(d=n_factors, scramble=True, seed=seed)
    return sampler.random(n=n_samples)

if __name__ == "__main__":
    sobol_design = create_sobol_design(factors, samples)

    # Array initialisieren, um das skalierte Design zu speichern
    sobol_scaled = np.zeros_like(sobol_design)

    # Faktor 1 skalieren (X-Position)
    sobol_scaled[:, 0] = sobol_design[:, 0] * (x_scale[1] - x_scale[0]) + x_scale[0]
    # Faktor 2 skalieren (Y-Position)
    sobol_scaled[:, 1] = sobol_design[:, 1] * (y_scale[1] - y_scale[0]) + y_scale[0]
    # Faktoren 3 bis 10 skalieren (Geometrievektoren)
    sobol_scaled[:, 2:] = sobol_design[:, 2:] * (geo_scale[1] - geo_scale[0]) + geo_scale[0]

    columns = ['X_Position', 'Y_Position'] + [f'Geom{i}' for i in range(1, 9)]
    sobol_df = pd.DataFrame(sobol_scaled, columns=columns)

    # Den DataFrame in eine Excel-Datei speichern
    sobol_df.to_excel('Sobol_Design.xlsx', index=False)

    # Die Daten auch als CSV-Datei speichern
    sobol_df.to_csv("Sobol_Design.csv", index=False)

    # Skalierung überprüfen, indem die ersten Stichproben ausgegeben werden
    np.set_printoptions(precision=2)  # bessere Lesbarkeit
    print("Erste 5 Stichproben von dem skalierten Sobol Design:")
    print(sobol_scaled[:5, :])

    print("Sobol Design wurde in 'Sobol_Design.xlsx' und 'Sobol_Design.csv' gespeichert.")