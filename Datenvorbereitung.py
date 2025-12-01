import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from numpy.linalg import norm
import os

"""# Dateien laden Model 1
df_taguchi = pd.read_excel('Daten/Taguchi_Einleger_Fliessfronten.xlsx').assign(Versuchsplan="Taguchi")
df_lhs    = pd.read_excel('Daten/LHS_Einleger_Fliessfronten.xlsx').assign(Versuchsplan="LHS")
df_sobol  = pd.read_excel('Daten/Sobol_Einleger_Fliessfronten.xlsx').assign(Versuchsplan="Sobol")
df_halton  = pd.read_excel('Daten/Halton_Einleger_Fliessfronten.xlsx').assign(Versuchsplan="Halton")"""

# Dateien laden Model 2
df_taguchi = pd.read_excel('Daten/Taguchi_Fliessfronten_Auslenkung.xlsx').assign(Versuchsplan="Taguchi")
df_lhs     = pd.read_excel('Daten/LHS_Fliessfronten_Auslenkung.xlsx').assign(Versuchsplan="LHS")
df_sobol   = pd.read_excel('Daten/Sobol_Fliessfronten_Auslenkung.xlsx').assign(Versuchsplan="Sobol")
df_halton  = pd.read_excel('Daten/Halton_Fliessfronten_Auslenkung.xlsx').assign(Versuchsplan="Halton")

OUT_DIR  = r"Getrennte_Daten"
N_TEST   = 391           # ~10% des Gesamtdatensets
PLAN_COL = "Versuchsplan"

os.makedirs(OUT_DIR, exist_ok=True)

# Alle Pläne zusammenführen
df_all = pd.concat([df_taguchi, df_lhs, df_sobol, df_halton], ignore_index=True)

"""# Ein- und Ausgabespalten für Modell 1
output_cols = ["Knoten_1", "Knoten_2", "Knoten_3"]
feat_cols = [c for c in df_all.columns if c not in ([PLAN_COL] + output_cols)]"""

# Ein- und Ausgabespalten für Modell 2
output_col = ["Auslenkung"]
feat_cols = [c for c in df_all.columns if c not in ([PLAN_COL] + output_col)]

# ---------- Duplikate erkennen ----------
df_all["__key__"] = df_all[feat_cols].round(12).astype(str).agg("|".join, axis=1)
grp = df_all.groupby("__key__").transform("size")
df_all["__dup"] = grp > 1
df_all = df_all.loc[~df_all["__dup"] | ~df_all.duplicated("__key__")].reset_index(drop=True)

# ---------- DUPLEX ----------
def build_duplex_holdout(df, n_test=N_TEST, plan_col=PLAN_COL, random_state=0):
    rng = np.random.default_rng(random_state)
    X = df[feat_cols].values  # nur Eingaben
    Xs = RobustScaler().fit_transform(X)

    plans, counts = np.unique(df[plan_col].values, return_counts=True)
    targets = {p: max(1, int(round(n_test * c/len(df)))) for p, c in zip(plans, counts)}

    eligible = ~df["__dup"].fillna(False)
    idx_all = np.where(eligible)[0]
    if len(idx_all) < n_test:
        raise ValueError("Zu wenige konfliktfreie Punkte für gewünschte Holdout-Größe.")

    start = idx_all[np.argmax(norm(Xs[idx_all], axis=1))]
    holdout = [start]
    taken = np.zeros(len(df), dtype=bool); taken[start] = True

    def can_take(i):
        pid = df.loc[i, plan_col]
        drawn = sum(df.loc[j, plan_col]==pid for j in holdout)
        return drawn < targets[pid]

    while len(holdout) < n_test:
        H = Xs[holdout]
        dmin = np.min(np.linalg.norm(Xs[:, None, :] - H[None, :, :], axis=2), axis=1)
        order = np.argsort(-dmin)
        chosen = None
        for i in order:
            if taken[i] or not eligible[i]: continue
            if can_take(i): chosen = i; break
        if chosen is None:
            for i in order:
                if not taken[i] and eligible[i]: chosen = i; break
        holdout.append(chosen); taken[chosen] = True

    return np.array(holdout, dtype=int)

# Holdout erzeugen
idx_holdout = build_duplex_holdout(df_all, n_test=N_TEST, plan_col=PLAN_COL, random_state=42)
df_holdout  = df_all.iloc[idx_holdout].copy()
df_pool_all = df_all.drop(df_all.index[idx_holdout]).copy()

# Speichern (Outputs bleiben enthalten)
df_holdout.drop(columns=["__key__","__dup"]).to_excel(os.path.join(OUT_DIR, "Holdout_fixed_Modell_2.xlsx"), index=False)

for plan_id, dfp in df_pool_all.groupby(PLAN_COL):
    dfp.drop(columns=["__key__","__dup"]).to_excel(os.path.join(OUT_DIR, f"Pool_{plan_id}_Modell_2.xlsx"), index=False)

print("Holdout_fixed_Modell_2.xlsx + Pool_<plan>_Moddell_2.xlsx sind gespeichert.")
