import re
import pandas as pd
from pathlib import Path

ROOT = Path("Ergebnisse_Teil_1")


def parse_study_name(study_name, model):
    m_plan = re.search(r"(Halton|LHS|Sobol|Taguchi)", study_name, re.I)
    plan = m_plan.group(1) if m_plan else None

    if model == 1:
        m_sub = re.search(r"Modell_1[._](\d)", study_name)
        sub = m_sub.group(1) if m_sub else None
        modell_label = f"1.{sub}" if sub else "1"
    else:
        sub = None
        modell_label = "2"

    m_split = re.search(r"Modell_\d(?:\.\d)?_(.*?)_seed", study_name)
    split = m_split.group(1) if m_split else None

    m_seed = re.search(r"seed[_-]?(\d+)", study_name)
    seed = int(m_seed.group(1)) if m_seed else None

    return plan, modell_label, sub, split, seed


def find_best_trial_in_study(study_file: Path):
    df = pd.read_excel(study_file)
    df_complete = df[df["state"] == "COMPLETE"]
    best_row = df_complete.loc[df_complete["value"].idxmin()]
    trial_number = int(best_row["number"])
    return trial_number, float(best_row["value"])


def extract_val_and_test_metrics(csv_path: Path):
    """
    Extrahiert RMSE/R² für Validation und Test aus metrics_*.csv-Dateien.
    Unterstützt Groß- und Kleinschreibung:
    - Dataset / dataset
    - RMSE / rmse
    - R2 / r2
    """

    result = {
        "r2_val": None,
        "rmse_val": None,
        "r2_test": None,
        "rmse_test": None,
    }

    if not csv_path.exists():
        return result

    df = pd.read_csv(csv_path)

    # --- Spalten case-insensitive finden ---
    cols_lower = {c.lower(): c for c in df.columns}

    # dataset column
    if "dataset" in cols_lower:
        dataset_col = cols_lower["dataset"]
    else:
        return result  # kein dataset → dann passt Datei nicht

    # rmse column
    if "rmse" in cols_lower:
        rmse_col = cols_lower["rmse"]
    else:
        return result

    # r2 column
    if "r2" in cols_lower:
        r2_col = cols_lower["r2"]
    else:
        return result

    # dataset-Werte normalisieren
    lower = df[dataset_col].astype(str).str.lower()

    # --- VALIDATION ---
    val_mask = lower.isin(["validation", "val"])
    if val_mask.any():
        row_val = df[val_mask].iloc[0]
        result["r2_val"] = float(row_val[r2_col])
        result["rmse_val"] = float(row_val[rmse_col])

    # --- TEST ---
    test_mask = lower.isin(["test"])
    if test_mask.any():
        row_test = df[test_mask].iloc[0]
        result["r2_test"] = float(row_test[r2_col])
        result["rmse_test"] = float(row_test[rmse_col])

    return result


def best_trials(model: int):

    rows = []

    for study_dir in sorted(ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        if model == 1 and "Modell_1" not in study_dir.name:
            continue
        if model == 2 and "Modell_2" not in study_dir.name:
            continue

        study_file = next((f for f in study_dir.glob("*.xlsx") if "Study" in f.name), None)
        if study_file is None:
            continue

        plan, modell_label, sub, split, seed = parse_study_name(study_dir.name, model)

        best_trial = find_best_trial_in_study(study_file)
        if best_trial is None:
            continue

        trial_number, best_value = best_trial

        metrics_dir = study_dir / "metrics"
        if not metrics_dir.exists():
            metrics_dir = study_dir

        # --- globale metrics_{trial}.csv ---
        metrics_main = metrics_dir / f"metrics_{trial_number}.csv"
        global_metrics = extract_val_and_test_metrics(metrics_main)

        from glob import glob

        def find_reeval_file(metrics_dir, trial_number, plan, model, sub):
            """
            Sucht die passende Reeval-Datei.
            Beispiel: metrics_70__Reeval_Halton_Modell_1.1.csv
            """

            # 1) Direktes korrektes Pattern (mit Punkt für Submodell)
            if model == 1 and sub is not None:
                direct = metrics_dir / f"metrics_{trial_number}__Reeval_{plan}_Modell_{model}.{sub}.csv"
            else:
                direct = metrics_dir / f"metrics_{trial_number}__Reeval_{plan}_Modell_{model}.csv"

            if direct.exists():
                return direct

            # 2) Fallback via glob (sehr tolerant)
            pattern = str(
                metrics_dir / f"metrics_{trial_number}__Reeval*{plan}*Modell*{model}*{sub if sub else ''}*.csv")
            matches = glob(pattern)

            if len(matches) > 0:
                return Path(matches[0])

            # 3) nichts gefunden
            return None

        # --- Reeval Datei finden ---
        metrics_reeval_path = find_reeval_file(metrics_dir, trial_number, plan, model, sub)

        if metrics_reeval_path is not None:
            reeval_metrics = extract_val_and_test_metrics(metrics_reeval_path)
        else:
            reeval_metrics = {"r2_val": None, "rmse_val": None, "r2_test": None, "rmse_test": None}

        rows.append({
            "Study": study_dir.name,
            "Versuchsplan": plan,
            "Modell": modell_label,
            "Split-Methode": split,
            "Seed": seed,
            "Bester_Trial": trial_number,

            # globale Metriken
            "bester_r2": global_metrics["r2_val"],
            "bester_rmse_val": best_value,
            "bester_r2_test": global_metrics["r2_test"],
            "bester_rmse_test": global_metrics["rmse_test"],

            # Reeval Test Werte
            "bester_r2_test_eigene_Daten": reeval_metrics["r2_test"],
            "bester_rmse_test_eigene_Daten": reeval_metrics["rmse_test"],
        })

    out_dir = Path("Ergebnisse_Summary_Reeval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"BestTrials_Modell_{model}.xlsx"

    pd.DataFrame(rows).to_excel(out_file, index=False)
    print(f"[OK] Übersicht gespeichert unter: {out_file}")


if __name__ == "__main__":
    best_trials(1)
    best_trials(2)