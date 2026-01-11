# -*- coding: utf-8 -*-
import re, importlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import math
import sys, types

def install_distutils_shim():
    if "distutils" in sys.modules:
        return

    distutils_stub = types.ModuleType("distutils")
    sys.modules["distutils"] = distutils_stub

    submodules = [
        "version", "errors", "util", "file_util", "dir_util", "spawn",
        "log", "archive_util", "text_file", "dep_util",
        "sysconfig", "ccompiler",
        "command", "command.build", "command.install", "command.build_ext"
    ]

    for sub in submodules:
        name = f"distutils.{sub}"
        mod = types.ModuleType(name)
        sys.modules[name] = mod

# vor TensorFlow importieren!
install_distutils_shim()

import tensorflow as tf

# ------------------- KONFIG -------------------
ROOT = Path("Ergebnisse_Teil_3")

DATA = Path("Getrennte_Daten")
SHEET = "All_Data"
SHEET_ALL = "All_Data"   # wird in upd_trial benutzt
AGG = "mean"

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
def parse_plan_and_submodel(study_name: str, model_nummer: int):
    """
    Plan ist fix:
      - Modell 1 -> Halton
      - Modell 2 -> LHS
    Split-Methode ist fix (nur Check):
      - Modell 1 -> SPXY
      - Modell 2 -> KS
    Submodell für Modell 1 wird weiterhin aus Modell_1.{1-3} geparst.
    """
    if model_nummer == 1:
        plan = "Halton"
        split_token = "SPXY"
        m_sub = re.search(r"Modell_1[._]([1-3])", study_name, re.I)
        if not m_sub:
            raise ValueError(f"Submodell in '{study_name}' nicht gefunden (erwartet Modell_1.[1-3]).")
        sub = m_sub.group(1)

    else:
        plan = "LHS"
        split_token = "KS"
        sub = None

    # Optionaler Plausibilitätscheck: Split-Token muss im Namen vorkommen
    # (hilft dir, falls ein falscher Ordner drin liegt)
    if re.search(rf"\b{split_token}\b", study_name, re.I) is None:
        print(f"[Hinweis] '{study_name}': erwarteter Split-Token '{split_token}' nicht im Namen gefunden.")

    return plan, sub


def load_normalization_and_holdout(plan: str, model_nummer: int, sub: str = None):
    """
    Lädt Pool/Holdout und gibt zurück:
        X_hold_norm, y_hold_true, norm_params

    Für Modell 1:
        - Dateien: Pool_{Plan}_Modell_1.xlsx, Holdout_{Plan}_fixed_Modell_1.xlsx
        - Eingaben: erste 10 Spalten
        - Ziel: je nach Submodell -4 / -3 / -2

    Für Modell 2 (Annahme – ggf. anpassen!):
        - Dateien: Pool_{Plan}_Modell_2.xlsx, Holdout_{Plan}_fixed_Modell_2.xlsx
        - Eingaben: alle Spalten außer letzte
        - Ziel: letzte Spalte
    """
    if model_nummer == 1:
        holdout_file = DATA / f"Holdout_{plan}_fixed_Modell_1.xlsx"
        pool_file    = DATA / f"Pool_{plan}_Modell_1.xlsx"
        if sub is None:
            raise ValueError("Für Modell 1 muss sub (1/2/3) gesetzt sein.")
    else:
        holdout_file = DATA / f"Holdout_{plan}_fixed_Modell_2.xlsx"
        pool_file    = DATA / f"Pool_{plan}_Modell_2.xlsx"

    if not holdout_file.exists():
        raise FileNotFoundError(f"Holdout-Datei nicht gefunden: {holdout_file}")
    if not pool_file.exists():
        raise FileNotFoundError(f"Pool-Datei nicht gefunden: {pool_file}")

    pool_df    = pd.read_excel(pool_file)
    holdout_df = pd.read_excel(holdout_file)

    if model_nummer == 1:
        # Eingaben: erste 10 Spalten
        pool_X    = pool_df.iloc[:, :10].astype(float)
        holdout_X = holdout_df.iloc[:, :10].astype(float)

        # Ausgabe-Spalte je nach Submodell
        ycol = SUBIDX_TO_YCOL.get(sub)
        if ycol is None:
            raise ValueError(f"Unbekanntes Submodell '{sub}'. Erwartet '1','2' oder '3'.")

        pool_y    = pool_df.iloc[:, ycol].astype(float).values.reshape(-1, 1)
        holdout_y = holdout_df.iloc[:, ycol].astype(float).values.reshape(-1, 1)

    else:
        # Modell 2:
        # - erste 10 Spalten: Inputs (X)
        # - vorletzte Spalte: Output (y)
        # - letzte Spalte: Name/String -> ignorieren
        if pool_df.shape[1] < 12:
            raise ValueError(
                f"Erwarte mindestens 12 Spalten in {pool_file} "
                f"(10 Inputs + 1 Output + 1 Name), gefunden: {pool_df.shape[1]}"
            )
        if holdout_df.shape[1] < 12:
            raise ValueError(
                f"Erwarte mindestens 12 Spalten in {holdout_file} "
                f"(10 Inputs + 1 Output + 1 Name), gefunden: {holdout_df.shape[1]}"
            )

        # erste 10 Spalten = Eingaben
        pool_X = pool_df.iloc[:, :10].astype(float)
        holdout_X = holdout_df.iloc[:, :10].astype(float)

        # vorletzte Spalte = Zielgröße
        pool_y = pool_df.iloc[:, -2].astype(float).values.reshape(-1, 1)
        holdout_y = holdout_df.iloc[:, -2].astype(float).values.reshape(-1, 1)

    # Min/Max aus dem POOL (wie im Training)
    min_x = pool_X.min().values.astype(float)
    max_x = pool_X.max().values.astype(float)
    min_y = float(pool_y.min())
    max_y = float(pool_y.max())

    # Normalisieren (wie im Training)
    denom_x = (max_x - min_x)
    denom_x[denom_x == 0] = 1.0  # zur Sicherheit
    X_hold_norm = (holdout_X.values - min_x) / denom_x

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
def upd_metrics(path_original: Path, r, R, plan, model_nummer: int, sub, trial):
    """
    Liest die ORIGINAL-metrics_{trial}.csv,
    aktualisiert darin NUR die Test-Zeile,
    schreibt aber eine NEUE Datei:

        Modell 1: metrics_{trial}__Reeval_{plan}_Modell_1.{sub}.csv
        Modell 2: metrics_{trial}__Reeval_{plan}_Modell_2.csv
    """
    try:
        df = pd.read_csv(path_original)
    except Exception as e:
        print(f"[Hinweis] Konnte Metrics-Datei {path_original} nicht lesen: {e}")
        return

    cols_lower = [c.strip().lower() for c in df.columns]
    df.columns = cols_lower

    if {"dataset", "rmse", "r2"}.issubset(df.columns):
        m = df["dataset"].astype(str).str.strip().str.lower().eq("test")
        if not m.any():
            df = pd.concat(
                [df, pd.DataFrame([{"dataset": "test", "rmse": np.nan, "r2": np.nan}])],
                ignore_index=True
            )
            m = df["dataset"].str.lower().eq("test")

        df.loc[m, ["rmse", "r2"]] = [r, R]
    else:
        df = pd.DataFrame({
            "dataset": ["test"],
            "rmse": [r],
            "r2": [R],
        })

    if model_nummer == 1:
        suffix = f"Modell_1.{sub}"
    else:
        suffix = "Modell_2"

    new_name = f"metrics_{trial}__Reeval_{plan}_{suffix}.csv"
    new_path = path_original.parent / new_name
    df.to_csv(new_path, index=False)
    print(f"[Neu] {new_name} in {new_path.parent} erzeugt.")


def upd_trial(path_original: Path, y_true, y_pred, plan, model_nummer: int, sub, trial):
    """
    - Liest die ORIGINAL-Trial_XXX.xlsx
    - Übernimmt alle Sheets
    - Schneidet plan- und modell-spezifisch alle Zeilen ab einer
      bestimmten Grenze weg (Taguchi/Sobol/LHS/Halton, Modell 1 & 2),
      um alte Eckpunkt-/Dummy-Zeilen am Ende zu entfernen.
    - Setzt in 'All_Data' die Spalten
        'True Values Test' und 'Predictions Test' neu (auf Basis der
      Reeval-Vorhersagen).
    - Schreibt NEUE Datei:

        Modell 1: Trial_{trial}__Reeval_{plan}_Modell_1.{sub}.xlsx
        Modell 2: Trial_{trial}__Reeval_{plan}_Modell_2.xlsx

    - Erzeugt einen Scatter-Chart True vs Predicted
    """

    try:
        xls = pd.ExcelFile(path_original)
        sheets = {name: xls.parse(name) for name in xls.sheet_names}
    except Exception as e:
        print(f"[Fehler] Konnte {path_original} nicht lesen: {e}")
        return

    if SHEET_ALL not in sheets:
        print(f"[Hinweis] {path_original.name}: Sheet '{SHEET_ALL}' nicht gefunden – nichts geändert.")
        return

    combined_df = sheets[SHEET_ALL].copy()

    """# -------------------------------------------------
    # Zeilen-Cutoffs pro Plan UND Modellnummer
    # (Excel-Zeilenangaben von dir, 1-basiert → pandas-Index 0-basiert)
    # -------------------------------------------------
    cutoffs = {
        "Taguchi": {1: 722, 2: 713},
        "Sobol":   {1: 820, 2: 811},
        "LHS":     {1: 786, 2: 778},
        "Halton":  {1: 802, 2: 788},
    }

    cutoff_row = cutoffs.get(plan, {}).get(model_nummer, None)
    if cutoff_row is not None and len(combined_df) > cutoff_row:
        # Excel-Zeile 722 → DataFrame-Zeilen 0..721 bleiben
        combined_df = combined_df.iloc[:cutoff_row].reset_index(drop=True)"""

    # Test-Spalten vorhanden?
    if "True Values Test" not in combined_df.columns or "Predictions Test" not in combined_df.columns:
        print(f"[Hinweis] {path_original.name}: 'True Values Test'/'Predictions Test' nicht gefunden – nichts geändert.")
        return

    y = np.array(y_true).reshape(-1)
    p = np.array(y_pred).reshape(-1)
    n = min(len(y), len(p))
    if n == 0:
        print(f"[Hinweis] {path_original.name}: Keine neuen Testdaten – nichts geändert.")
        return

    # Test-Spalten neu setzen
    combined_df["True Values Test"] = np.nan
    combined_df["Predictions Test"] = np.nan
    combined_df.loc[:n - 1, "True Values Test"] = y[:n]
    combined_df.loc[:n - 1, "Predictions Test"] = p[:n]

    # updated Sheet zurückschreiben
    sheets[SHEET_ALL] = combined_df

    # neue Datei schreiben + Chart neu aufbauen
    if model_nummer == 1:
        suffix = f"Modell_1.{sub}"
    else:
        suffix = "Modell_2"

    new_name = f"Trial_{trial}__Reeval_{plan}_{suffix}.xlsx"
    new_path = path_original.parent / new_name

    with pd.ExcelWriter(new_path, engine="xlsxwriter") as writer:
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

        start_row = 1
        end_row   = len(combined_df)
        end_row_test = n

        for dataset, color, (col_x, col_y) in [
            ("Train",      "#95BB20", (col_true_train, col_pred_train)),
            ("Validation", "#717E86", (col_true_val,   col_pred_val)),
            ("Test",       "#00354E", (col_true_test,  col_pred_test)),
        ]:
            if dataset == "Test":
                last_row = start_row + end_row_test - 1
            else:
                last_row = start_row + end_row - 1

            chart_all.add_series({
                "name":       f"{dataset} Data",
                "categories": [SHEET_ALL, start_row, col_x, last_row, col_x],
                "values":     [SHEET_ALL, start_row, col_y, last_row, col_y],
                "marker": {
                    "type":   "circle",
                    "size":   6,
                    "fill":   {"color": color},
                    "border": {"none": True},
                },
            })

        max_value = math.ceil(np.nanmax([
            combined_df["True Values Train"].max(),
            combined_df["True Values Validation"].max(),
            combined_df["True Values Test"].max(),
            combined_df["Predictions Train"].max(),
            combined_df["Predictions Validation"].max(),
            combined_df["Predictions Test"].max(),
        ]))

        col_chart_x = 17
        col_chart_y = 18

        worksheet_all.write(0, col_chart_x, "Chart_X")
        worksheet_all.write(0, col_chart_y, "Chart_Y")

        dummy_points = [
            (0, 0),
            (max_value, max_value),
            (max_value, 0),
            (max_value, max_value),
            (0, max_value),
            (max_value, max_value),
        ]

        for i, (cx, cy) in enumerate(dummy_points, start=1):
            worksheet_all.write(i, col_chart_x, cx)
            worksheet_all.write(i, col_chart_y, cy)

        chart_all.add_series({
            "name": "45° Linie",
            "categories": [SHEET_ALL, 1, col_chart_x, 2, col_chart_x],
            "values":     [SHEET_ALL, 1, col_chart_y, 2, col_chart_y],
            "line": {"color": "black", "dash_type": "dash", "width": 1.0},
            "marker": {"type": "none"},
        })

        chart_all.add_series({
            "name": None,
            "categories": [SHEET_ALL, 3, col_chart_x, 4, col_chart_x],
            "values":     [SHEET_ALL, 3, col_chart_y, 4, col_chart_y],
            "line": {"color": "black", "width": 1.0},
            "marker": {"type": "none"},
        })

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

        chart_all.set_plotarea({
            "border": {"color": "black", "width": 0.5},
            "fill":   {"color": "white"},
        })
        chart_all.set_chartarea({
            "border": {"color": "black", "width": 0.5},
            "fill":   {"color": "white"},
        })

        worksheet_all.insert_chart("J2", chart_all)

    print(f"[OK] {new_name}: Testdaten aktualisiert, Zeilen-Cutoff angewendet, Chart neu erzeugt.")

def save_pairs(pairs_dir: Path, study, trial, y_true, y_pred):
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": np.arange(len(y_true)),
        "y_true": np.array(y_true).reshape(-1),
        "y_pred": np.array(y_pred).reshape(-1)
    }).to_excel(
        pairs_dir / f"{study}__trial_{trial}__{t}__TestPairs.xlsx",
        index=False
    )


def write_summary(rows, summary_path: Path):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if summary_path.exists():
        df_old = pd.read_excel(summary_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_excel(summary_path, index=False)


# --- Re-Evaluierung je Study (intern) ---
def _reevaluate_study(study: Path, load_model, model_nummer: int):
    study_name = study.name
    plan, sub = parse_plan_and_submodel(study_name, model_nummer)

    # dynamische Ergebnisordner pro Plan & Modell
    OUT   = Path("Ergebnisse_Summary_Reeval_Datenreduktion") / f"Reeval_{plan}_Modell_{model_nummer}"
    PAIRS = OUT / "Test-Predict-Pairs"
    OUT.mkdir(parents=True, exist_ok=True)
    PAIRS.mkdir(parents=True, exist_ok=True)

    SUMMARY_XLSX = OUT / f"Reeval_{plan}_Modell_{model_nummer}_summary.xlsx"

    X_hold_norm, y_true, norm_params = load_normalization_and_holdout(plan, model_nummer, sub)
    min_x, max_x, min_y, max_y = norm_params

    models_dir = study / "models"
    if not models_dir.is_dir():
        return

    rows = []

    for mdl in sorted(models_dir.glob("*.keras")):
        m = re.search(r"_([0-9]+)\.keras$", mdl.name)
        if not m:
            continue
        trial = m.group(1)

        model = load_model(mdl.as_posix())
        n_in = int(model.input_shape[-1])

        X_use_norm = X_hold_norm[:, :n_in] if X_hold_norm.shape[1] >= n_in else X_hold_norm
        if X_hold_norm.shape[1] != n_in:
            print(f"[Hinweis] {study_name}: X({X_hold_norm.shape[1]}) -> Modell erwartet {n_in}, "
                  f"nutze {X_use_norm.shape[1]}.")

        @tf.function(reduce_retracing=True)
        def predict_step(x):
            return model(x, training=False)

        P_norm = predict_step(tf.constant(X_use_norm, dtype=tf.float32)).numpy()

        if isinstance(P_norm, list):
            P_norm = np.concatenate(
                [p if p.ndim > 1 else p.reshape(-1, 1) for p in P_norm],
                axis=1
            )
        if P_norm.ndim == 1:
            P_norm = P_norm.reshape(-1, 1)

        y_pred = unnormalize_y(P_norm, min_y, max_y)

        r = rmse(y_true, y_pred)
        R = r2(y_true, y_pred)

        save_pairs(PAIRS, study_name, trial, y_true, y_pred)

        if model_nummer == 1:
            sub_label = f"1.{sub}"
        else:
            sub_label = "2"

        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "study": study_name,
            "trial": int(trial),
            "plan": plan,
            "submodel": sub_label,
            "n_samples": len(y_true),
            "rmse_test": r,
            "r2_test": R,
            "model_path": mdl.as_posix()
        })

        met = study / "metrics"
        if met.is_dir():
            c = list(met.glob(f"*{trial}*.csv"))
            if c:
                upd_metrics(c[0], r, R, plan, model_nummer, sub, trial)

        tx = study / f"Trial_{trial}.xlsx"
        if not tx.exists():
            tx = (study / "metrics") / f"Trial_{trial}.xlsx"
        if tx.exists():
            upd_trial(tx, y_true, y_pred, plan, model_nummer, sub, trial)

        print(f"[OK] {study_name} Trial {trial}: RMSE_test={r:.4f}  R2_test={R:.4f}")

    write_summary(rows, SUMMARY_XLSX)
    if rows:
        print(f"\nSummary geschrieben nach: {SUMMARY_XLSX}")
        print(f"Test-Paare gespeichert in: {PAIRS}")


# ---------- Öffentliche Funktion: reevaluate(1) / reevaluate(2) ----------
def reevaluate(model_nummer: int, base_dir: str = "Ergebnisse_Teil_3"):
    """
    Reevaluierung für Modell 1 oder 2 in Ergebnisse_Teil_3
    - Modell 1:  Study_15_10_2025_Halton_Modell_1.{sub}_Halton_SPXY_{CBIR|RegMix}_Holdout_seed_*
    - Modell 2:  Study_15_10_2025_Halton_Modell_2_Halton_KS_{CBIR|RegMix}_Holdout_seed_*
    """
    if model_nummer not in (1, 2):
        raise ValueError("model_nummer muss 1 oder 2 sein.")

    LM = _lm()
    root = Path(base_dir)

    if model_nummer == 1:
        patt = re.compile(
            r"Study_15_10_2025_.*Modell_1[._][1-3].*SPXY.*(CBIR|RegMix).*Holdout_seed_",
            re.I
        )
    else:
        patt = re.compile(
            r"Study_15_10_2025_.*Modell_2.*KS.*(CBIR|RegMix).*Holdout_seed_",
            re.I
        )

    for s in sorted(root.iterdir()):
        if s.is_dir() and patt.search(s.name):  # <- search statt match
            _reevaluate_study(s, LM, model_nummer)


def main():
    # Wenn du das Skript direkt ausführst, kannst du hier z.B. nur Modell 1 laufen lassen:
    # reevaluate(1)
    # oder beide:
    for m in (1, 2):
        try:
            reevaluate(m, base_dir="Ergebnisse_Teil_3")
        except Exception as e:
            print(f"[Warnung] Fehler bei Modell {m}: {e}")


if __name__ == "__main__":
    main()