# -*- coding: utf-8 -*-
import re, sys, types, importlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import math


# ------------------- KONFIG -------------------
ROOT = Path("Ergebnisse_Teil_1")

# Nur Modell_1.x-Studien für Halton/LHS/Sobol/Taguchi
ONLY = re.compile(r"(Halton|LHS|Sobol|Taguchi).*Modell_1(\.|_)", re.I)

DATA = Path("Getrennte_Daten")
HOLDOUT_FILE = DATA / "Holdout_fixed_Modell_1.xlsx"   # Holdout wie im Training
SHEET = "All_Data"
AGG = "mean"

OUT = Path("Ergebnisse_Summary_Reeval") / "Reeval_Modell_1"
PAIRS = OUT / "Test-Predict-Pairs"
OUT.mkdir(parents=True, exist_ok=True)
PAIRS.mkdir(parents=True, exist_ok=True)
SUMMARY_XLSX = OUT / "Reeval_Summary_Modell_1.xlsx"

# Mapping Submodell → y-Spalte (wie von dir: 1.1=-4, 1.2=-3, 1.3=-2)
SUBIDX_TO_YCOL = {"1": -4, "2": -3, "3": -2}

# --- TF-Loader mit minimalem 'distutils'-Shim (für Py3.12) ---
def _lm():
    if "distutils" not in sys.modules:
        sys.modules["distutils"] = types.ModuleType("distutils")
        for sub in ("version", "errors", "util", "spawn"):
            sys.modules[f"distutils.{sub}"] = types.ModuleType(f"distutils.{sub}")
    tf = importlib.import_module("tensorflow")
    return lambda p: tf.keras.models.load_model(p, compile=False)

# --------- Hilfsfunktionen zum Parsen & Laden ---------
def parse_plan_and_submodel(study_name: str):
    """
    Extrahiert Plan (Halton/LHS/Sobol/Taguchi) und Submodell-Index '1','2','3'
    aus dem Studiennamen, z.B.:
    Study_15_10_2025_Halton_Modell_1.3_KS_Holdout_seed_0
        -> plan='Halton', sub='3'
    """
    m_plan = re.search(r"(Halton|LHS|Sobol|Taguchi)", study_name, re.I)
    if not m_plan:
        raise ValueError(f"Plan (Halton/LHS/Sobol/Taguchi) in '{study_name}' nicht gefunden.")
    plan = m_plan.group(1)

    m_sub = re.search(r"Modell_1[._](\d)", study_name, re.I)
    if not m_sub:
        raise ValueError(f"Submodell in '{study_name}' nicht gefunden (erwartet Modell_1.[1-3]).")
    sub = m_sub.group(1)  # '1', '2' oder '3'
    return plan, sub


def load_normalization_and_holdout(plan: str, sub: str):
    """
    Lädt Pool_<Plan>_Modell_1.xlsx und Holdout_fixed_Modell_1.xlsx,
    berechnet Min/Max für X und y (aus Pool) und gibt zurück:
        X_hold_norm, y_hold_true, norm_params
    wobei:
        X_hold_norm: normalisiertes Holdout-X (wie im Training),
        y_hold_true: unnormierte Zielwerte aus Holdout,
        norm_params: (min_x, max_x, min_y, max_y)
    """
    pool_file = DATA / f"Pool_{plan}_Modell_1.xlsx"
    if not pool_file.exists():
        raise FileNotFoundError(f"Pool-Datei nicht gefunden: {pool_file}")

    # Pool & Holdout laden
    pool_df = pd.read_excel(pool_file)
    holdout_df = pd.read_excel(HOLDOUT_FILE)

    # Eingaben: erste 10 Spalten
    pool_X = pool_df.iloc[:, :10].astype(float)
    holdout_X = holdout_df.iloc[:, :10].astype(float)

    # Ausgabe-Spalte je nach Submodell
    ycol = SUBIDX_TO_YCOL.get(sub)
    if ycol is None:
        raise ValueError(f"Unbekanntes Submodell '{sub}'. Erwartet '1','2' oder '3'.")

    pool_y = pool_df.iloc[:, ycol].astype(float).values.reshape(-1, 1)
    holdout_y = holdout_df.iloc[:, ycol].astype(float).values.reshape(-1, 1)

    # Min/Max aus dem POOL (wie im Training)
    min_x = pool_X.min().values.astype(float)
    max_x = pool_X.max().values.astype(float)
    min_y = float(pool_y.min())
    max_y = float(pool_y.max())

    # Normalisieren (wie im Training)
    denom_x = (max_x - min_x)
    denom_x[denom_x == 0] = 1.0  # zur Sicherheit
    X_hold_norm = (holdout_X.values - min_x) / denom_x

    # (y_hold würde im Training auch normiert werden, aber wir vergleichen in Originaleinheiten)
    norm_params = (min_x, max_x, min_y, max_y)

    return X_hold_norm, holdout_y, norm_params


def unnormalize_y(y_norm, min_y, max_y):
    return y_norm * (max_y - min_y) + min_y

# --- Metriken ---
def rmse(y, p):
    e = np.sqrt(np.mean((y - p) ** 2, axis=0))
    return float(np.mean(e) if AGG == "mean" else np.sum(e))

def r2(y, p):
    ssr = np.sum((y - p) ** 2, axis=0)
    sst = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 1.0 - ssr / sst
    v = np.where(np.isfinite(v), v, -np.inf)
    return float(np.mean(v) if AGG == "mean" else np.sum(v))


# --- Artefakte aktualisieren ---
def upd_metrics(path: Path, r, R):
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    low = [c.strip().lower() for c in df.columns]
    if {"dataset", "rmse", "r2"}.issubset(low):
        df.columns = low
        m = df["dataset"].astype(str).str.strip().str.lower().eq("test")
        if not m.any():
            df = pd.concat([df, pd.DataFrame([{"dataset": "test", "rmse": np.nan, "r2": np.nan}])],
                           ignore_index=True)
            m = df["dataset"].str.lower().eq("test")
        df.loc[m, ["rmse", "r2"]] = [r, R]
    else:
        if "RMSE_Test" in df.columns:
            df.loc[df.index[-1], "RMSE_Test"] = r
        if "R2_Test" in df.columns:
            df.loc[df.index[-1], "R2_Test"] = R
    df.to_csv(path, index=False)

SHEET_ALL = "All_Data"

SHEET_ALL = "All_Data"

def upd_trial(path: Path, y_true, y_pred, plan: str):
    """
    - Liest die existierende Trial_XXX.xlsx
    - Setzt in 'All_Data' die Spalten
        'True Values Test' und 'Predictions Test' zunächst auf NaN
      und schreibt dann NUR die eigenen Reeval-Testdaten in die ersten n Zeilen.
    - Schneidet plan-spezifisch alles ab bestimmter Zeile ab
      (Taguchi, Sobol, LHS, Halton, jeweils Modell 1).
    - Löscht Spalten 7–10 (Index 6–9).
    - Erzeugt einen Scatter-Chart mit Train/Val/Test-Punkten.
    - 45°-Linie + vertikale/horizontale Linie kommen aus den
      Spalten 'Chart_X' / 'Chart_Y' rechts von den Daten.
    """

    # 1) Vorhandene Sheets einlesen
    try:
        xls = pd.ExcelFile(path)
        sheets = {name: xls.parse(name) for name in xls.sheet_names}
    except Exception as e:
        print(f"[Fehler] Konnte {path} nicht lesen: {e}")
        return

    if SHEET_ALL not in sheets:
        print(f"[Hinweis] {path.name}: Sheet '{SHEET_ALL}' nicht gefunden – nichts geändert.")
        return

    combined_df = sheets[SHEET_ALL].copy()


    """# Abschneide-Grenzen pro Plan (Modell 1)
    cutoffs = {
        "Taguchi": 722,
        "Sobol":   820,
        "LHS":     786,
        "Halton":  802,
    }

    cutoff_row = cutoffs.get(plan, len(combined_df))
    if len(combined_df) > cutoff_row:
        combined_df = combined_df.iloc[:cutoff_row].reset_index(drop=True)

    # --- SPALTEN 7–10 (Index 6–9) LÖSCHEN ---
    if len(combined_df.columns) >= 10:
        cols_to_drop = combined_df.columns[6:10]
        combined_df = combined_df.drop(columns=cols_to_drop)"""

    # 2) Test-Spalten prüfen
    if "True Values Test" not in combined_df.columns or "Predictions Test" not in combined_df.columns:
        print(f"[Hinweis] {path.name}: 'True Values Test'/'Predictions Test' nicht gefunden – nichts geändert.")
        return

    y = np.array(y_true).reshape(-1)
    p = np.array(y_pred).reshape(-1)

    n = min(len(y), len(p))
    if n == 0:
        print(f"[Hinweis] {path.name}: Keine Testdaten – nichts geändert.")
        return

    # 3) Test-Spalten leeren und neue Testdaten schreiben
    combined_df["True Values Test"] = np.nan
    combined_df["Predictions Test"] = np.nan
    combined_df.loc[:n-1, "True Values Test"] = y[:n]
    combined_df.loc[:n-1, "Predictions Test"] = p[:n]

    # 4) Ggf. alte Eckpunkt-Heckzeilen am Tabellenende entfernen (Train 0/0, max/0, 0/max, max/max)
    if "True Values Train" in combined_df.columns and "Predictions Train" in combined_df.columns:
        tv_train = combined_df["True Values Train"].to_numpy(dtype=float)
        pv_train = combined_df["Predictions Train"].to_numpy(dtype=float)

        with np.errstate(all="ignore"):
            max_val_data = np.nanmax([tv_train, pv_train])
        if not np.isfinite(max_val_data):
            max_val_data = 0.0

        corner_candidates = [
            (0.0, 0.0),
            (max_val_data, max_val_data),
            (0.0, max_val_data),
            (max_val_data, 0.0),
        ]

        def is_corner_pair(x, y, tol=1e-9):
            if np.isnan(x) or np.isnan(y):
                return False
            for cx, cy in corner_candidates:
                if abs(x - cx) < tol and abs(y - cy) < tol:
                    return True
            return False

        last_real_idx = len(combined_df) - 1
        while last_real_idx >= 0 and is_corner_pair(tv_train[last_real_idx], pv_train[last_real_idx]):
            last_real_idx -= 1

        if last_real_idx < len(combined_df) - 1:
            combined_df = combined_df.iloc[:last_real_idx+1].reset_index(drop=True)

    # 5) Hilfsspalten Chart_X / Chart_Y (rechts anhängen, keine neuen Zeilen)
    if "Chart_X" not in combined_df.columns:
        combined_df["Chart_X"] = np.nan
    if "Chart_Y" not in combined_df.columns:
        combined_df["Chart_Y"] = np.nan

    real_slice = combined_df
    max_value = math.ceil(np.nanmax([
        real_slice.get("True Values Train", pd.Series([0])).max(),
        real_slice.get("True Values Validation", pd.Series([0])).max(),
        real_slice.get("True Values Test", pd.Series([0])).max(),
        real_slice.get("Predictions Train", pd.Series([0])).max(),
        real_slice.get("Predictions Validation", pd.Series([0])).max(),
        real_slice.get("Predictions Test", pd.Series([0])).max(),
    ]))

    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0

    dummy_points = [
        (0.0, 0.0),
        (max_value, max_value),
        (max_value, 0.0),
        (max_value, max_value),
        (0.0, max_value),
        (max_value, max_value),
    ]

    n_dummy = min(len(dummy_points), len(combined_df))
    for i in range(n_dummy):
        combined_df.at[i, "Chart_X"] = dummy_points[i][0]
        combined_df.at[i, "Chart_Y"] = dummy_points[i][1]

    # 6) zurück ins Sheet-Dict
    sheets[SHEET_ALL] = combined_df

    # 7) Datei neu schreiben & Chart neu aufbauen
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=name, index=False)

        workbook = writer.book
        worksheet_all = writer.sheets[SHEET_ALL]

        chart_all = workbook.add_chart({"type": "scatter"})

        col_true_train = combined_df.columns.get_loc("True Values Train")
        col_pred_train = combined_df.columns.get_loc("Predictions Train")
        col_true_val   = combined_df.columns.get_loc("True Values Validation")
        col_pred_val   = combined_df.columns.get_loc("Predictions Validation")
        col_true_test  = combined_df.columns.get_loc("True Values Test")
        col_pred_test  = combined_df.columns.get_loc("Predictions Test")
        col_chart_x    = combined_df.columns.get_loc("Chart_X")
        col_chart_y    = combined_df.columns.get_loc("Chart_Y")

        start_row    = 1
        last_row_all = start_row + len(combined_df) - 1
        last_row_test = start_row + n - 1

        # Train / Val / Test
        for dataset, color, (col_x, col_y, last_row) in [
            ("Train",      "#95BB20", (col_true_train, col_pred_train, last_row_all)),
            ("Validation", "#717E86", (col_true_val,   col_pred_val,   last_row_all)),
            ("Test",       "#00354E", (col_true_test,  col_pred_test,  last_row_test)),
        ]:
            chart_all.add_series({
                "name":       f"{dataset} Data",
                "categories": [SHEET_ALL, start_row, col_x, last_row, col_x],
                "values":     [SHEET_ALL, start_row, col_y, last_row, col_y],
                "marker": {
                    "type":  "circle",
                    "size":  6,
                    "fill":   {"color": color},
                    "border": {"none": True},
                },
            })

        # 45°-Linie
        if n_dummy >= 2:
            chart_all.add_series({
                "name": "45° Linie",
                "categories": [SHEET_ALL, 1, col_chart_x, 2, col_chart_x],
                "values":     [SHEET_ALL, 1, col_chart_y, 2, col_chart_y],
                "line": {"color": "black", "dash_type": "dash", "width": 1.0},
                "marker": {"type": "none"},
            })

        # Vertikale Linie x = max_value
        if n_dummy >= 4:
            chart_all.add_series({
                "name": None,
                "categories": [SHEET_ALL, 3, col_chart_x, 4, col_chart_x],
                "values":     [SHEET_ALL, 3, col_chart_y, 4, col_chart_y],
                "line": {"color": "black", "width": 1.0},
                "marker": {"type": "none"},
            })

        # Horizontale Linie y = max_value
        if n_dummy >= 6:
            chart_all.add_series({
                "name": None,
                "categories": [SHEET_ALL, 5, col_chart_x, 6, col_chart_x],
                "values":     [SHEET_ALL, 5, col_chart_y, 6, col_chart_y],
                "line": {"color": "black", "width": 1.0},
                "marker": {"type": "none"},
            })

        chart_all.set_title({
            "name": "True vs Predicted Values",
            "name_font": {"size": 14, "bold": True, "color": "black"},
        })

        chart_all.set_x_axis({
            "name": "True Values [mm]",
            "name_font": {"size": 12, "bold": False, "color": "black"},
            "num_font":  {"size": 10, "color": "black"},
            "line": {"color": "black", "width": 1.0},
            "major_gridlines": {
                "visible": True,
                "line": {"color": "black", "width": 0.5},
            },
            "min": 0,
            "max": max_value,
        })

        chart_all.set_y_axis({
            "name": "Predicted Values [mm]",
            "name_font": {"size": 12, "bold": False, "color": "black"},
            "num_font":  {"size": 10, "color": "black"},
            "line": {"color": "black", "width": 1.0},
            "major_gridlines": {
                "visible": True,
                "line": {"color": "black", "width": 0.5},
            },
            "min": 0,
            "max": max_value,
        })

        chart_all.set_plotarea({"border": {"color": "black", "width": 0.5}, "fill": {"color": "white"}})
        chart_all.set_chartarea({"border": {"color": "black", "width": 0.5}, "fill": {"color": "white"}})

        worksheet_all.insert_chart("J2", chart_all)

    print(f"[OK] {path.name}: Testdaten aktualisiert, Chart mit Hilfsspalten neu erzeugt.")


def save_pairs(study, trial, y_true, y_pred):
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame({
        "index": np.arange(len(y_true)),
        "y_true": np.array(y_true).reshape(-1),
        "y_pred": np.array(y_pred).reshape(-1)
    }).to_excel(
        PAIRS / f"{study}__trial_{trial}__{t}__TestPairs.xlsx",
        index=False
    )


def write_summary(rows):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if SUMMARY_XLSX.exists():
        df_old = pd.read_excel(SUMMARY_XLSX)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_excel(SUMMARY_XLSX, index=False)


# --- Re-Evaluierung je Study ---
def reevaluate(study: Path, rows, load_model):
    study_name = study.name
    plan, sub = parse_plan_and_submodel(study_name)

    # Daten & Normalisierung wie im Training
    X_hold_norm, y_true, norm_params = load_normalization_and_holdout(plan, sub)
    min_x, max_x, min_y, max_y = norm_params

    models_dir = study / "models"
    if not models_dir.is_dir():
        return

    for mdl in sorted(models_dir.glob("*.keras")):
        m = re.search(r"_([0-9]+)\.keras$", mdl.name)
        if not m:
            continue
        trial = m.group(1)

        model = load_model(mdl.as_posix())
        n_in = int(model.input_shape[-1])

        # ggf. auf n_in kürzen, falls weniger Eingänge genutzt wurden
        X_use_norm = X_hold_norm[:, :n_in] if X_hold_norm.shape[1] >= n_in else X_hold_norm
        if X_hold_norm.shape[1] != n_in:
            print(f"[Hinweis] {study_name}: X({X_hold_norm.shape[1]}) -> Modell erwartet {n_in}, "
                  f"nutze {X_use_norm.shape[1]}.")

        # Vorhersage im normierten Raum
        P_norm = model.predict(X_use_norm, verbose=0)
        if isinstance(P_norm, list):
            P_norm = np.concatenate(
                [p if p.ndim > 1 else p.reshape(-1, 1) for p in P_norm],
                axis=1
            )
        if P_norm.ndim == 1:
            P_norm = P_norm.reshape(-1, 1)

        # Wieder unnormalisieren (wie im Training)
        y_pred = unnormalize_y(P_norm, min_y, max_y)

        # RMSE & R2 im Originalmaßstab
        r = rmse(y_true, y_pred)
        R = r2(y_true, y_pred)

        save_pairs(study_name, trial, y_true, y_pred)
        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "study": study_name,
            "trial": int(trial),
            "plan": plan,
            "submodel": f"1.{sub}",
            "n_samples": len(y_true),
            "rmse_test": r,
            "r2_test": R,
            "model_path": mdl.as_posix()
        })

        # metrics_*.csv anpassen (falls vorhanden)
        met = study / "metrics"
        if met.is_dir():
            c = list(met.glob(f"*{trial}*.csv"))
            if c:
                upd_metrics(c[0], r, R)

        # Trial_*.xlsx anpassen (falls vorhanden)
        tx = study / f"Trial_{trial}.xlsx"
        if not tx.exists():
            tx = (study / "metrics") / f"Trial_{trial}.xlsx"
        if tx.exists():
            upd_trial(tx, y_true, y_pred, plan)

        print(f"[OK] {study_name} Trial {trial}: RMSE_test={r:.4f}  R2_test={R:.4f}")


def main():
    rows = []
    LM = _lm()
    for s in sorted(ROOT.iterdir()):
        if s.is_dir() and ONLY.search(s.name):
            reevaluate(s, rows, LM)
    write_summary(rows)
    if rows:
        print(f"\nSummary geschrieben nach: {SUMMARY_XLSX}")
        print(f"Test-Paare gespeichert in: {PAIRS}")

if __name__ == "__main__":
    main()