import numpy as np
from scipy.stats import qmc
from scipy.spatial import distance
import pandas as pd

# Anzahl der Faktoren und Stichproben
factors = 10
samples = 980

x_scale = [-350, 350] # X-Position
y_scale = [-150, 150] # Y-Position
geo_scale = [0, 700] # Geometrievektoren
seed = 42  # Seed für Reproduzierbarkeit

def create_lhs_maximin(n_factors, n_samples, n_iter=20):
    """Erzeugt ein Latin Hypercube Sampling (LHS) Design mit maximin-Kriterium."""
    best_lhs = None
    best_min_dist = -np.inf
    for _ in range(n_iter):
        sampler = qmc.LatinHypercube(d=n_factors, scramble=True, seed=seed)
        sample = sampler.random(n=n_samples)
        dists = distance.pdist(sample)
        min_dist = np.min(dists)
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_lhs = sample
    return best_lhs


if __name__ == "__main__":
    lhs_design = create_lhs_maximin(factors, samples)

    # Array initialisieren, um das skalierte Design zu speichern
    lhs_scaled = np.zeros_like(lhs_design)

    # Faktor 1 skalieren (X-Position)
    lhs_scaled[:, 0] = lhs_design[:, 0] * (x_scale[1] - x_scale[0]) + x_scale[0]
    # Faktor 2 skalieren (Y-Position)
    lhs_scaled[:, 1] = lhs_design[:, 1] * (y_scale[1] - y_scale[0]) + y_scale[0]
    # Faktoren 3 bis 10 skalieren (Geometrievektoren)
    lhs_scaled[:, 2:] = lhs_design[:, 2:] * (geo_scale[1] - geo_scale[0]) + geo_scale[0]

    columns = ['X_Position', 'Y_Position'] + [f'Geom{i}' for i in range(1, 9)]
    lhs_df = pd.DataFrame(lhs_scaled, columns=columns)

    # Den DataFrame in eine Excel-Datei speichern
    lhs_df.to_excel('LHS_Design2.xlsx', index=False)

    # Skalierung überprüfen, indem die ersten Stichproben ausgegeben werden
    np.set_printoptions(precision=2)  # bessere Lesbarkeit
    print("Erste 5 Stichproben von dem skalierten LHS Design:")
    print(lhs_scaled[:5, :])

    # Die Daten auch als CSV-Datei speichern
    np.savetxt("Daten/LHS_Design.csv", lhs_scaled, delimiter=",", comments='')

    print("LHS Design wurde in 'LHS_Design.xlsx' und 'LHS_Design.csv' gespeichert.")