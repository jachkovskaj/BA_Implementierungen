# cleanup_trials_modell_2.py
# -*- coding: utf-8 -*-
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------- KONFIG -------------------
ROOT = Path("Ergebnisse_Teil_1")
SHEET_ALL = "All_Data"

# Nur Modell_2-Studien für Halton/LHS/Sobol/Taguchi
ONLY_M2 = re.compile(r"(Halton|LHS|Sobol|Taguchi).*Modell_2(\.|_)", re.I)

# Abschneide-Grenzen für Modell 2
CUTROWS_M2 = {
    "Taguchi": 713,
    "Sobol":   811,
    "LHS":     778,
    "Halton":  788,
}


def parse_plan(study_name: str) -> str:
    """
    Extrahiert den Plan (Halton/LHS/Sobol/Taguchi) aus dem Studiennamen.
    """
    m_plan = re.search(r"(Halton|LHS|Sobol|Taguchi)", study_name, re.I)
    if not m_plan:
        raise ValueError(f"Plan (Halton/LHS/Sobol/Taguchi) in '{study_name}' nicht gefunden.")
    return m_plan.group(1)


def build_chart_for_df(workbook, worksheet_all, df: pd.DataFrame, max_value: float):
    """
    Baut den Scatter-Chart so auf, wie in deinem Modell-1-upd_trial:
    - Train / Validation / Test mit gleichen Farben & Markern
    - 45°-Linie + vertikale + horizontale Linie aus Chart_X / Chart_Y
    - Achsen, Grid, Layout identisch
    """

    chart_all = workbook.add_chart({"type": "scatter"})

    # Spalten-Indices
    col_true_train = df.columns.get_loc("True Values Train")
    col_pred_train = df.columns.get_loc("Predictions Train")
    col_true_val   = df.columns.get_loc("True Values Validation")
    col_pred_val   = df.columns.get_loc("Predictions Validation")
    col_true_test  = df.columns.get_loc("True Values Test")
    col_pred_test  = df.columns.get_loc("Predictions Test")
    col_chart_x    = df.columns.get_loc("Chart_X")
    col_chart_y    = df.columns.get_loc("Chart_Y")

    start_row    = 1  # Excel: zweite Zeile, erste Datenzeile
    last_row_all = start_row + len(df) - 1

    # Letzte Test-Zeile anhand nicht-NaN in True Values Test
    tv_test = df["True Values Test"].to_numpy(dtype=float)
    valid_idx = np.where(~np.isnan(tv_test))[0]
    if valid_idx.size > 0:
        last_idx_test = valid_idx[-1]
        last_row_test = start_row + last_idx_test
    else:
        # Falls alles NaN: Test-Serie sehr kurz halten
        last_row_test = start_row

    # Train / Val / Test Serien
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

    # Dummy-Punkte-Anzahl prüfen
    n_dummy = df["Chart_X"].notna().sum()

    # 45°-Linie
    if n_dummy >= 2:
        chart_all.add_series({
            "name": "45° Linie",
            "categories": [SHEET_ALL, 1, col_chart_x, 2, col_chart_x],
            "values":     [SHEET_ALL, 1, col_chart_y, 2, col_chart_y],
            "line": {"color": "black", "dash_type": "dash", "width": 1.0},
            "marker": {"type": "none"},
        })

    # Vertikale Linie x = max_value (Punkte 3 & 4)
    if n_dummy >= 4:
        chart_all.add_series({
            "name": None,
            "categories": [SHEET_ALL, 3, col_chart_x, 4, col_chart_x],
            "values":     [SHEET_ALL, 3, col_chart_y, 4, col_chart_y],
            "line": {"color": "black", "width": 1.0},
            "marker": {"type": "none"},
        })

    # Horizontale Linie y = max_value (Punkte 5 & 6)
    if n_dummy >= 6:
        chart_all.add_series({
            "name": None,
            "categories": [SHEET_ALL, 5, col_chart_x, 6, col_chart_x],
            "values":     [SHEET_ALL, 5, col_chart_y, 6, col_chart_y],
            "line": {"color": "black", "width": 1.0},
            "marker": {"type": "none"},
        })

    # Achsen & Layout – wie in deinem Skript
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


def cleanup_trial_modell2(path: Path, plan: str):
    """
    Bereinigt eine Trial_*.xlsx-Datei für Modell 2:
    - schneidet ab plan-spezifischer Zeile (Taguchi, Sobol, LHS, Halton)
    - löscht Spalten 7–10
    - baut Chart neu auf wie im Modell-1-upd_trial
    """

    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[Fehler] Konnte {path} nicht lesen: {e}")
        return

    if SHEET_ALL not in xls.sheet_names:
        print(f"[Hinweis] {path.name}: Sheet '{SHEET_ALL}' nicht gefunden – übersprungen.")
        return

    sheets = {name: xls.parse(name) for name in xls.sheet_names}
    df = sheets[SHEET_ALL].copy()

    # 1) plan-spezifisch Zeilen abschneiden
    cutoff_excel = CUTROWS_M2.get(plan, None)
    if cutoff_excel is not None:
        # Excel-Zeilen → DataFrame-Zeilen:
        # Excel 1 = Header, Excel 2 = df.index 0
        # "ab Zeile cutoff_excel löschen" => bis Zeile (cutoff_excel - 1) behalten
        # Anzahl DataFrame-Zeilen = (cutoff_excel - 2)
        n_keep = max(cutoff_excel - 2, 0)
        if len(df) > n_keep:
            df = df.iloc[:n_keep].reset_index(drop=True)
    else:
        # Fallback – nichts abschneiden
        n_keep = len(df)

    # 2) Spalten 7–10 (Index 6–9) löschen
    if len(df.columns) >= 10:
        df = df.drop(columns=df.columns[6:10])

    # 3) Chart_X / Chart_Y anlegen oder leeren
    if "Chart_X" not in df.columns:
        df["Chart_X"] = np.nan
    else:
        df["Chart_X"] = np.nan  # Zur Sicherheit alles neu setzen

    if "Chart_Y" not in df.columns:
        df["Chart_Y"] = np.nan
    else:
        df["Chart_Y"] = np.nan

    # max_value aus realen Daten
    max_value = math.ceil(np.nanmax([
        df.get("True Values Train", pd.Series([0])).max(),
        df.get("True Values Validation", pd.Series([0])).max(),
        df.get("True Values Test", pd.Series([0])).max(),
        df.get("Predictions Train", pd.Series([0])).max(),
        df.get("Predictions Validation", pd.Series([0])).max(),
        df.get("Predictions Test", pd.Series([0])).max(),
    ]))
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0

    # Dummy-Punkte wie im Modell-1-Skript
    dummy_points = [
        (0.0, 0.0),
        (max_value, max_value),
        (max_value, 0.0),
        (max_value, max_value),
        (0.0, max_value),
        (max_value, max_value),
    ]
    n_dummy = min(len(dummy_points), len(df))
    for i in range(n_dummy):
        df.at[i, "Chart_X"] = dummy_points[i][0]
        df.at[i, "Chart_Y"] = dummy_points[i][1]

    # aktualisiertes All_Data zurück ins Sheet-Dict
    sheets[SHEET_ALL] = df

    # 4) Datei neu schreiben + Chart wie in Modell 1 aufbauen
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for name, df_sheet in sheets.items():
                df_sheet.to_excel(writer, sheet_name=name, index=False)

            workbook = writer.book
            worksheet_all = writer.sheets[SHEET_ALL]

            build_chart_for_df(workbook, worksheet_all, df, max_value)

        print(f"[OK] {path.name}: bereinigt & Chart neu erstellt.")
    except Exception as e:
        print(f"[Fehler beim Schreiben von {path}: {e}")


def main():
    for study in sorted(ROOT.iterdir()):
        if not study.is_dir():
            continue
        if not ONLY_M2.search(study.name):
            continue

        try:
            plan = parse_plan(study.name)
        except ValueError as e:
            print(f"[Hinweis] {e}")
            continue

        print(f"\n=== Bereinige Trials für Modell 2: {study.name} (Plan: {plan}) ===")

        # Trial-Dateien im Study-Root
        for trial_xlsx in study.glob("Trial_*.xlsx"):
            cleanup_trial_modell2(trial_xlsx, plan)

        # Trial-Dateien im metrics/-Ordner (falls vorhanden)
        metrics_dir = study / "metrics"
        if metrics_dir.is_dir():
            for trial_xlsx in metrics_dir.glob("Trial_*.xlsx"):
                cleanup_trial_modell2(trial_xlsx, plan)


if __name__ == "__main__":
    main()