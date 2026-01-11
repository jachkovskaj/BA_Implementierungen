import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# ============================================================
# Konfiguration
# ============================================================
base_dir = "Ergebnisse_Teil_2"
output_dir = "Ergebnisplots_Visualization_Metriken_kombiniert_Datenaufteilung"
os.makedirs(output_dir, exist_ok=True)

SPLIT_ORDER = ["DUPLEX", "KS", "SPlit", "SPXY"]  # so wie du es nebeneinander willst
SPLIT_COLOR_MAP = {
    "KS":     "#eef7ff",
    "DUPLEX": "#fff3e0",
    "SPlit":  "#e8f5e9",
    "SPXY":   "#fce4ec",
}
DEFAULT_BG = "#ffffff"

# Modell 1 -> nur Halton, Modell 2 -> nur LHS (wie bei dir)
PLAN_RESTRICTION = {1: "Halton", 2: "LHS"}

# ============================================================
# Hilfsfunktionen
# ============================================================
def save_plot(fig, filename, subfolder):
    save_path = os.path.join(output_dir, subfolder)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

def read_metrics(metrics_path):
    """
    Liest eine metrics_*.csv ein und gibt ein Dict zurück:
    {
      'train':      {'rmse': ..., 'r2': ...},
      'validation': {'rmse': ..., 'r2': ...},
      'test':       {'rmse': ..., 'r2': ...}
    }
    """
    m = pd.read_csv(metrics_path)
    m.columns = [str(c).strip().lower() for c in m.columns]
    out = {
        "train": {"rmse": np.nan, "r2": np.nan},
        "validation": {"rmse": np.nan, "r2": np.nan},
        "test": {"rmse": np.nan, "r2": np.nan},
    }

    # Standard-Format: dataset/rmse/r2
    if {"dataset", "rmse", "r2"}.issubset(m.columns):
        norm = {
            "train": "train", "training": "train",
            "valid": "validation", "validation": "validation", "val": "validation",
            "test": "test", "testing": "test"
        }
        m["dataset"] = m["dataset"].astype(str).str.strip().str.lower().map(norm)
        g = (m.dropna(subset=["dataset"])
               .groupby("dataset", sort=False)[["rmse", "r2"]]
               .agg("last"))
        for ds in out:
            if ds in g.index:
                out[ds]["rmse"] = float(g.loc[ds, "rmse"])
                out[ds]["r2"]   = float(g.loc[ds, "r2"])
        return out

    # Alternative Spaltennamen (Fallback)
    def get(col):
        return m[col].iloc[-1] if col in m.columns and not m[col].empty else np.nan

    for ds in out:
        out[ds]["rmse"] = get(f"rmse_{ds}")
        out[ds]["r2"]   = get(f"r2_{ds}")
    return out

def is_modell(study_name: str, n: int) -> bool:
    # Verhindert Modell_11 etc.
    return re.search(rf"_Modell_{n}(?!\d)", study_name) is not None

def get_plan(study_name: str) -> str:
    m = re.search(r"_(Halton|LHS|Sobol|Taguchi)_", study_name, re.I)
    if not m:
        return "Other"
    token = m.group(1).lower()
    plan_map = {"halton": "Halton", "lhs": "LHS", "sobol": "Sobol", "taguchi": "Taguchi"}
    return plan_map.get(token, m.group(1))

def get_split_method(study_name: str) -> str:
    m = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name, re.I)
    if not m:
        return "Other"
    token = m.group(1)
    # Case normalisieren (wegen re.I)
    split_map = {"ks": "KS", "duplex": "DUPLEX", "split": "SPlit", "spxy": "SPXY"}
    return split_map.get(token.lower(), token)

def get_seed_int(study_name: str) -> int:
    m = re.search(r"seed_(\d+)", study_name, re.I)
    return int(m.group(1)) if m else 999999

def get_submodel(study_name: str):
    # nur Modell 1: Modell_1.1 / Modell_1.2 / Modell_1.3
    m = re.search(r"Modell_1\.(\d)", study_name)
    return m.group(1) if m else None

def get_model_str(study_name: str) -> str:
    m = re.search(r"Modell_([\d.]+)", study_name)
    return m.group(1) if m else "?"

def get_model_sortkey(model_str: str):
    parts = str(model_str).split(".")
    try:
        major = int(parts[0])
    except Exception:
        major = 999
    minor = 0
    if len(parts) > 1:
        try:
            minor = int(parts[1])
        except Exception:
            minor = 0
    return (major, minor)

def shorten_name(study_name: str) -> str:
    # wie du es willst: "1.2_KS_42" bzw "2_DUPLEX_0"
    mdl  = get_model_str(study_name)
    sp   = get_split_method(study_name)
    seed = get_seed_int(study_name)
    return f"{mdl}_{sp}_{seed}"

def _shade_background(ax, split_names):
    for i, s in enumerate(split_names):
        ax.axvspan(i - 0.5, i + 0.5, color=SPLIT_COLOR_MAP.get(s, DEFAULT_BG), alpha=0.7, zorder=0)

def _safe_labels(labels):
    return [("?" if (s is None or (isinstance(s, float) and np.isnan(s))) else str(s)) for s in labels]

def _scatter_by_index(ax, labels, y, **kwargs):
    x = np.arange(len(labels))
    y = np.asarray(y, dtype=float)
    ax.scatter(x, y, **kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels(_safe_labels(labels), rotation=90, ha="center")

def aggregate_per_split(values, splits):
    """
    Gibt Dicts zurück:
      mean[s] = Mittelwert über alle Studies mit split==s
      std[s]  = Std (ddof=1) über alle Studies mit split==s
    für s in SPLIT_ORDER
    """
    mean = {}
    std = {}
    for s in SPLIT_ORDER:
        vals = np.array([v for v, sp in zip(values, splits) if sp == s], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            mean[s] = np.nan
            std[s] = np.nan
        else:
            mean[s] = float(vals.mean())
            std[s] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, std

def find_best_trial_in_study(study_xlsx: str):
    df = pd.read_excel(study_xlsx)
    if "state" not in df.columns or "value" not in df.columns:
        return None
    df_complete = df[df["state"] == "COMPLETE"]
    if df_complete.empty:
        return None
    best_row = df_complete.loc[df_complete["value"].idxmin()]
    if "number" in df_complete.columns:
        trial = int(best_row["number"])
    elif "trial_id" in df_complete.columns:
        trial = int(best_row["trial_id"])
    else:
        return None
    return trial

def find_reeval_metrics(metrics_dir: str, trial: int, n: int, sub: str | None, fixed_plan: str):
    """
    Sucht Reeval-Metrics CSV robust:
    1) ZUERST strikt: metrics_{trial}__Reeval_{fixed_plan}_...Modell_...
    2) Dann fallback: metrics_{trial}__Reeval_*...Modell_...
    """
    if n == 1 and sub is not None:
        strict = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_{fixed_plan}_*Modell_1.{sub}*.csv")
        loose  = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_*Modell_1.{sub}*.csv")
    else:
        strict = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_{fixed_plan}_*Modell_{n}*.csv")
        loose  = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_*Modell_{n}*.csv")

    matches = sorted(glob(strict))
    if matches:
        return matches[0]

    matches = sorted(glob(loose))
    if matches:
        return matches[0]

    return None

# ============================================================
# Hauptauswertung (analog zu Teil 1)
# ============================================================
def auswertung_modell(n: int):
    fixed_plan = PLAN_RESTRICTION.get(n, None)
    records = []

    for study in sorted(os.listdir(base_dir)):
        study_path = os.path.join(base_dir, study)
        if not os.path.isdir(study_path):
            continue
        if not is_modell(study, n):
            continue

        # Plan je Modell FEST (nicht "plan" aus Name parsen)
        if fixed_plan is not None:
            if f"_{fixed_plan}_" not in study:
                continue

        split = get_split_method(study)
        if split not in SPLIT_ORDER:
            continue

        # Study Excel finden
        xlsx_candidates = [f for f in os.listdir(study_path) if f.endswith(".xlsx")]
        if not xlsx_candidates:
            continue
        study_xlsx = os.path.join(study_path, sorted(xlsx_candidates)[-1])

        best_trial = find_best_trial_in_study(study_xlsx)
        if best_trial is None:
            continue

        metrics_dir = os.path.join(study_path, "metrics")
        if not os.path.isdir(metrics_dir):
            continue

        # 1) GLOBAL metrics_{trial}.csv
        global_csv = os.path.join(metrics_dir, f"metrics_{best_trial}.csv")
        if not os.path.isfile(global_csv):
            # Fallback: nur NICHT-Reeval CSVs nehmen
            cand = sorted([
                f for f in os.listdir(metrics_dir)
                if f.endswith(".csv")
                   and str(best_trial) in f
                   and "__Reeval_" not in f
            ])
            if cand:
                global_csv = os.path.join(metrics_dir, cand[0])
            else:
                # letzter fallback: irgendeine nicht-Reeval csv
                all_csv = sorted([
                    f for f in os.listdir(metrics_dir)
                    if f.endswith(".csv") and "__Reeval_" not in f
                ])
                if not all_csv:
                    continue
                global_csv = os.path.join(metrics_dir, all_csv[-1])

        metrics_global = read_metrics(global_csv)

        # 2) EIGENE Daten (Reeval)
        sub = get_submodel(study) if n == 1 else None
        reeval_csv = find_reeval_metrics(metrics_dir, best_trial, n, sub, fixed_plan) if fixed_plan else None
        metrics_own = read_metrics(reeval_csv) if reeval_csv and os.path.isfile(reeval_csv) else None

        # records
        mdl_str = get_model_str(study)
        records.append({
            "label": shorten_name(study),
            "split": split,
            "model_key": get_model_sortkey(mdl_str),
            "seed": get_seed_int(study),

            # global
            "rmse_train": metrics_global["train"]["rmse"],
            "rmse_val": metrics_global["validation"]["rmse"],
            "rmse_test_global": metrics_global["test"]["rmse"],

            "r2_train": metrics_global["train"]["r2"],
            "r2_val": metrics_global["validation"]["r2"],
            "r2_test_global": metrics_global["test"]["r2"],

            # own (nur test sinnvoll, train/val bleiben NaN)
            "rmse_test_own": (metrics_own["test"]["rmse"] if metrics_own else np.nan),
            "r2_test_own":   (metrics_own["test"]["r2"]   if metrics_own else np.nan),
        })

    if not records:
        print(f"Keine Daten für Modell {n} gefunden.")
        return

    # ✅ Sortierung wie bei dir: Split-Block -> Modell -> Seed
    split_rank = {s: i for i, s in enumerate(SPLIT_ORDER)}
    records.sort(key=lambda r: (split_rank.get(r["split"], 999), r["model_key"], r["seed"]))

    study_names = [r["label"] for r in records]
    split_names = [r["split"] for r in records]

    rmse_train = np.array([r["rmse_train"] for r in records], float)
    rmse_val   = np.array([r["rmse_val"] for r in records], float)
    rmse_test_global = np.array([r["rmse_test_global"] for r in records], float)
    rmse_test_own    = np.array([r["rmse_test_own"] for r in records], float)

    r2_train = np.array([r["r2_train"] for r in records], float)
    r2_val   = np.array([r["r2_val"] for r in records], float)
    r2_test_global = np.array([r["r2_test_global"] for r in records], float)
    r2_test_own    = np.array([r["r2_test_own"] for r in records], float)

    subfolder = f"Modell_{n}"

    # ============================================================
    # Plot 1: RMSE Train/Val/Test(global) + Test(eigene)
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_background(ax, split_names)

    _scatter_by_index(ax, study_names, rmse_train, marker="o", label="Train (global)", zorder=2, color="#95BB20")
    _scatter_by_index(ax, study_names, rmse_val,   marker="o", label="Validation (global)", zorder=2, color="#00354E")
    _scatter_by_index(ax, study_names, rmse_test_global, marker="o", label="Test (global)", zorder=2, color="#717E86")
    _scatter_by_index(ax, study_names, rmse_test_own,    marker="o", label="Test (eigene Daten)", zorder=3, color="#0098AD")

    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE – Modell {n} – global vs. eigene Testdaten (Plan fix: {PLAN_RESTRICTION.get(n,'alle')})")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8))
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_{n}.png", subfolder)

    # ============================================================
    # Plot 2: R² Train/Val/Test(global) + Test(eigene)
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_background(ax, split_names)

    _scatter_by_index(ax, study_names, r2_train, marker="o", label="Train (global)", zorder=2, color="#95BB20")
    _scatter_by_index(ax, study_names, r2_val,   marker="o", label="Validation (global)", zorder=2, color="#00354E")
    _scatter_by_index(ax, study_names, r2_test_global, marker="o", label="Test (global)", zorder=2, color="#717E86")
    _scatter_by_index(ax, study_names, r2_test_own,    marker="o", label="Test (eigene Daten)", zorder=3, color="#0098AD")

    ax.set_ylabel("R²")
    ax.set_title(f"R² – Modell {n} – global vs. eigene Testdaten (Plan fix: {PLAN_RESTRICTION.get(n,'alle')})")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_{n}.png", subfolder)

    # ============================================================
    # Plot 3: RMSE (Train/Val/Test global/Test eigene) pro Split – gruppierte Balken
    # ============================================================
    mean_tr, std_tr = aggregate_per_split(rmse_train, split_names)
    mean_v, std_v = aggregate_per_split(rmse_val, split_names)
    mean_g, std_g = aggregate_per_split(rmse_test_global, split_names)
    mean_o, std_o = aggregate_per_split(rmse_test_own, split_names)

    x = np.arange(len(SPLIT_ORDER))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - 1.5 * width, [mean_tr[s] for s in SPLIT_ORDER], width,
           yerr=[std_tr[s] for s in SPLIT_ORDER], capsize=4,
           label="Train (global)", color="#95BB20", edgecolor="black")

    ax.bar(x - 0.5 * width, [mean_v[s] for s in SPLIT_ORDER], width,
           yerr=[std_v[s] for s in SPLIT_ORDER], capsize=4,
           label="Validation (global)", color="#00354E", edgecolor="black")

    ax.bar(x + 0.5 * width, [mean_g[s] for s in SPLIT_ORDER], width,
           yerr=[std_g[s] for s in SPLIT_ORDER], capsize=4,
           label="Test (global)", color="#717E86", edgecolor="black")

    ax.bar(x + 1.5 * width, [mean_o[s] for s in SPLIT_ORDER], width,
           yerr=[std_o[s] for s in SPLIT_ORDER], capsize=4,
           label="Test (eigene Daten)", color="#0098AD", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_ylim(top=20)
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE (Train/Val/Test global/Test eigene) pro Aufteilungsmethode (Modell {n}.*)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_plot(fig, f"RMSE_TrainValTest_by_split_Modell_{n}.png", subfolder)

    # ============================================================
    # Plot 4: R² (Train/Val/Test global/Test eigene) pro Split – gruppierte Balken
    # ============================================================
    mean_tr, std_tr = aggregate_per_split(r2_train, split_names)
    mean_v, std_v = aggregate_per_split(r2_val, split_names)
    mean_g, std_g = aggregate_per_split(r2_test_global, split_names)
    mean_o, std_o = aggregate_per_split(r2_test_own, split_names)

    x = np.arange(len(SPLIT_ORDER))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - 1.5 * width, [mean_tr[s] for s in SPLIT_ORDER], width,
           yerr=[std_tr[s] for s in SPLIT_ORDER], capsize=4,
           label="Train (global)", color="#95BB20", edgecolor="black")

    ax.bar(x - 0.5 * width, [mean_v[s] for s in SPLIT_ORDER], width,
           yerr=[std_v[s] for s in SPLIT_ORDER], capsize=4,
           label="Validation (global)", color="#00354E", edgecolor="black")

    ax.bar(x + 0.5 * width, [mean_g[s] for s in SPLIT_ORDER], width,
           yerr=[std_g[s] for s in SPLIT_ORDER], capsize=4,
           label="Test (global)", color="#717E86", edgecolor="black")

    ax.bar(x + 1.5 * width, [mean_o[s] for s in SPLIT_ORDER], width,
           yerr=[std_o[s] for s in SPLIT_ORDER], capsize=4,
           label="Test (eigene Daten)", color="#0098AD", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_ylabel("R²")
    ax.set_ylim(bottom=-0.9)
    ax.set_title(f"R² (Train/Val/Test global/Test eigene) pro Aufteilungsmethode (Modell {n}.*)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")

    plt.tight_layout()
    save_plot(fig, f"R2_TrainValTest_by_split_Modell_{n}.png", subfolder)

    print(f"✔ Modell {n} ausgewertet → {os.path.join(output_dir, subfolder)}")


if __name__ == "__main__":
    #auswertung_modell(1)
    auswertung_modell(2)