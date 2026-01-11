import re
import pandas as pd
from pathlib import Path
from glob import glob

ROOT = Path("Ergebnisse_Teil_3")

REDUCTION_ORDER = ["CBIR", "RegMix"]

# Fix je Modell
MODEL_FIX = {
    1: {"plan": "Halton", "split": "SPXY"},
    2: {"plan": "LHS",    "split": "KS"},
}


def parse_study_name(study_name: str, model: int):
    # Versuchsplan
    m_plan = re.search(r"_(Halton|LHS|Sobol|Taguchi)_", study_name, re.I)
    plan = m_plan.group(1) if m_plan else None
    if plan:
        plan = plan[0].upper() + plan[1:].lower()

    # Modell / Submodell
    if model == 1:
        m_sub = re.search(r"Modell_1[._]([1-3])\b", study_name, re.I)
        sub = m_sub.group(1) if m_sub else None
        modell_label = f"1.{sub}" if sub else "1"
    else:
        sub = None
        modell_label = "2"

    # Split
    m_split = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name, re.I)
    split = m_split.group(1) if m_split else None
    if split:
        split_map = {"ks": "KS", "duplex": "DUPLEX", "split": "SPlit", "spxy": "SPXY"}
        split = split_map.get(split.lower(), split)

    # Datenreduktion
    m_red = re.search(r"_(CBIR|RegMix)_", study_name, re.I)
    red = m_red.group(1) if m_red else None
    if red:
        red = "CBIR" if red.lower() == "cbir" else "RegMix"

    # CONF (nur bei CBIR relevant, aber tolerant)
    m_conf = re.search(r"_CONF_(\d+)", study_name, re.I)
    conf = int(m_conf.group(1)) if m_conf else None

    # Seed
    m_seed = re.search(r"seed[_-]?(\d+)", study_name, re.I)
    seed = int(m_seed.group(1)) if m_seed else None

    return plan, modell_label, sub, split, seed, red, conf


def find_best_trial_in_study(study_file: Path):
    df = pd.read_excel(study_file)
    df_complete = df[df["state"] == "COMPLETE"]
    if df_complete.empty:
        return None

    best_row = df_complete.loc[df_complete["value"].idxmin()]

    if "number" in df_complete.columns:
        trial_number = int(best_row["number"])
    elif "trial_id" in df_complete.columns:
        trial_number = int(best_row["trial_id"])
    else:
        trial_number = int(best_row.name)

    best_value = float(best_row["value"])
    return trial_number, best_value


def extract_val_and_test_metrics(csv_path: Path):
    """
    Extrahiert RMSE/R² für Validation und Test aus metrics_*.csv-Dateien.
    Unterstützt dataset/rmse/r2 case-insensitiv.
    """
    result = {"r2_val": None, "rmse_val": None, "r2_test": None, "rmse_test": None}

    if not csv_path or not Path(csv_path).exists():
        return result

    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}

    if not {"dataset", "rmse", "r2"}.issubset(cols_lower):
        return result

    dataset_col = cols_lower["dataset"]
    rmse_col = cols_lower["rmse"]
    r2_col = cols_lower["r2"]

    lower = df[dataset_col].astype(str).str.strip().str.lower()

    # Validation
    val_mask = lower.isin(["validation", "val", "valid"])
    if val_mask.any():
        row_val = df[val_mask].iloc[0]
        result["r2_val"] = float(row_val[r2_col])
        result["rmse_val"] = float(row_val[rmse_col])

    # Test
    test_mask = lower.isin(["test", "testing"])
    if test_mask.any():
        row_test = df[test_mask].iloc[0]
        result["r2_test"] = float(row_test[r2_col])
        result["rmse_test"] = float(row_test[rmse_col])

    return result


def find_metrics_csv(metrics_dir: Path, trial_number: int):
    """
    Sucht in /metrics nach metrics_*.csv:
      1) metrics_{trial}.csv (exakt)
      2) metrics_{trial}*.csv (z.B. falls Suffix dran hängt)
      3) fallback: irgendein metrics_*.csv (größte Datei als Heuristik)
    """
    metrics_dir = Path(metrics_dir)

    c1 = metrics_dir / f"metrics_{trial_number}.csv"
    if c1.exists():
        return c1

    candidates = sorted(glob(str(metrics_dir / f"metrics_{trial_number}*.csv")))
    if candidates:
        candidates.sort()
        return Path(candidates[-1])

    candidates = [Path(p) for p in glob(str(metrics_dir / "metrics_*.csv"))]
    if candidates:
        # nimm die größte Datei als fallback (oft „vollständigste“)
        candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        return candidates[0]

    return None


def best_trials(model: int):
    if model not in (1, 2):
        raise ValueError("model muss 1 oder 2 sein.")

    fixed_plan = MODEL_FIX[model]["plan"]
    fixed_split = MODEL_FIX[model]["split"]

    rows = []

    for study_dir in sorted(ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        # Modellfilter auf Ordnernamen
        if model == 1 and "Modell_1" not in study_dir.name:
            continue
        if model == 2 and "Modell_2" not in study_dir.name:
            continue

        # Study-Excel finden
        study_file = next((f for f in study_dir.glob("*.xlsx") if "Study" in f.name), None)
        if study_file is None:
            continue

        plan, modell_label, sub, split, seed, red, conf = parse_study_name(study_dir.name, model)

        # ✅ Fixierungen
        if plan != fixed_plan:
            continue
        if split != fixed_split:
            continue

        # ✅ Nur Reduktionsmethoden
        if red not in REDUCTION_ORDER:
            continue

        best_trial = find_best_trial_in_study(study_file)
        if best_trial is None:
            continue
        trial_number, best_value = best_trial

        metrics_dir = study_dir / "metrics"
        if not metrics_dir.exists():
            # falls mal alles im Study-Root liegt
            metrics_dir = study_dir

        metrics_csv = find_metrics_csv(metrics_dir, trial_number)
        if metrics_csv is None:
            print(f"[Hinweis] Keine metrics_*.csv in: {metrics_dir}")
            continue

        m = extract_val_and_test_metrics(metrics_csv)

        rows.append({
            "Study": study_dir.name,
            "Versuchsplan_fix": plan,
            "Split_fix": split,
            "Reduktion": red,
            "CONF": conf,
            "Modell": modell_label,
            "Seed": seed,
            "Bester_Trial": trial_number,
            "Best_value": best_value,

            # Metrics aus /metrics/metrics_*.csv
            "r2_val": m["r2_val"],
            "rmse_val": m["rmse_val"],
            "r2_test": m["r2_test"],
            "rmse_test": m["rmse_test"],
            "metrics_file": str(metrics_csv),
        })

    out_dir = Path("Ergebnisse_Summary_Datenreduktion")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"BestTrials_Modell_{model}_Datenreduktion.xlsx"

    df_out = pd.DataFrame(rows)

    # Sortierung: Reduktion -> Modell -> Seed -> CONF
    red_rank = {r: i for i, r in enumerate(REDUCTION_ORDER)}

    def modell_key(m):
        parts = str(m).split(".")
        major = int(parts[0]) if parts[0].isdigit() else 999
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (major, minor)

    if not df_out.empty:
        df_out["__red_rank"] = df_out["Reduktion"].map(red_rank).fillna(999).astype(int)
        df_out["__modell_key"] = df_out["Modell"].apply(modell_key)
        df_out["__seed"] = df_out["Seed"].fillna(999999).astype(int)
        df_out["__conf"] = df_out["CONF"].fillna(999999).astype(int)

        df_out = df_out.sort_values(
            by=["__red_rank", "__modell_key", "__seed", "__conf"],
            ascending=True
        ).drop(columns=["__red_rank", "__modell_key", "__seed", "__conf"])

    df_out.to_excel(out_file, index=False)
    print(f"[OK] Übersicht gespeichert unter: {out_file}")


if __name__ == "__main__":
    best_trials(1)
    best_trials(2)