import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# ============================================================
# Konfiguration
# ============================================================
base_dir = "Ergebnisse_Teil_3"
output_dir = "Ergebnisplots_Visualization_Metriken_kombiniert_Datenreduktion"
os.makedirs(output_dir, exist_ok=True)

REDUCTION_ORDER = ["CBIR", "RegMix"]
REDUCTION_COLOR_MAP = {
    "CBIR":  "#eef7ff",   # blau
    "RegMix":"#fff3e0",   # gelb
}
DEFAULT_BG = "#ffffff"

# Fixierungen
PLAN_FIX  = {1: "Halton", 2: "LHS"}
SPLIT_FIX = {1: "SPXY",   2: "KS"}

# ============================================================
# Helpers
# ============================================================
def save_plot(fig, filename, subfolder):
    save_path = os.path.join(output_dir, subfolder)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

def read_metrics(metrics_path):
    """
    Liest RMSE/R² für Train/Validation/Test.
    Unterstützt:
      - dataset/rmse/r2 (Zeilenformat)
      - rmse_train/r2_train/... (Spaltenformat)
    """
    m = pd.read_csv(metrics_path)
    m.columns = [str(c).strip().lower() for c in m.columns]

    out = {
        "train": {"rmse": np.nan, "r2": np.nan},
        "validation": {"rmse": np.nan, "r2": np.nan},
        "test": {"rmse": np.nan, "r2": np.nan},
    }

    if {"dataset", "rmse", "r2"}.issubset(m.columns):
        norm = {
            "train": "train", "training": "train",
            "valid": "validation", "validation": "validation", "val": "validation",
            "test": "test", "testing": "test",
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

    def get(col):
        return m[col].iloc[-1] if col in m.columns and not m[col].empty else np.nan

    for ds in out:
        out[ds]["rmse"] = get(f"rmse_{ds}")
        out[ds]["r2"]   = get(f"r2_{ds}")
    return out

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

def get_seed_int(study_name: str) -> int:
    m = re.search(r"seed_(\d+)", study_name, re.I)
    return int(m.group(1)) if m else 999999

def get_conf_int(study_name: str):
    m = re.search(r"_CONF_(\d+)", study_name, re.I)
    return int(m.group(1)) if m else None

def get_reduction_method(study_name: str) -> str:
    m = re.search(r"_(CBIR|RegMix)_", study_name, re.I)
    if not m:
        return "Other"
    token = m.group(1).lower()
    return "CBIR" if token == "cbir" else "RegMix" if token == "regmix" else m.group(1)

def get_plan(study_name: str) -> str:
    m = re.search(r"_(Halton|Sobol|Taguchi|LHS)_", study_name, re.IGNORECASE)
    if not m:
        return "Other"

    token = m.group(1).lower()
    if token == "lhs":
        return "LHS"
    if token == "halton":
        return "Halton"
    if token == "sobol":
        return "Sobol"
    if token == "taguchi":
        return "Taguchi"
    return m.group(1)

def get_split_method(study_name: str) -> str:
    m = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name, re.I)
    if not m:
        return "Other"
    token = m.group(1).lower()
    split_map = {"ks": "KS", "duplex": "DUPLEX", "split": "SPlit", "spxy": "SPXY"}
    return split_map.get(token, m.group(1))

def get_submodel(study_name: str):
    m = re.search(r"Modell_1\.(\d)", study_name)
    return m.group(1) if m else None

def pick_study_excel(study_path: str, study_name: str):
    preferred = os.path.join(study_path, f"{study_name}.xlsx")
    if os.path.isfile(preferred):
        return preferred
    xlsx_files = [f for f in os.listdir(study_path) if f.lower().endswith(".xlsx")]
    if not xlsx_files:
        return None
    xlsx_files.sort(key=lambda fn: os.path.getsize(os.path.join(study_path, fn)), reverse=True)
    return os.path.join(study_path, xlsx_files[0])

def find_metrics_csv(metrics_dir: str, trial: int):
    """
    Sucht NUR in /metrics nach metrics_*.csv (keine Reeval).
    """
    if not os.path.isdir(metrics_dir):
        return None

    direct = os.path.join(metrics_dir, f"metrics_{trial}.csv")
    if os.path.isfile(direct):
        return direct

    cands = [f for f in os.listdir(metrics_dir)
             if f.lower().endswith(".csv")
             and f.lower().startswith("metrics_")
             and str(trial) in f
             and "__reeval" not in f.lower()]
    if cands:
        cands.sort()
        return os.path.join(metrics_dir, cands[-1])

    cands = [f for f in os.listdir(metrics_dir)
             if f.lower().endswith(".csv")
             and f.lower().startswith("metrics_")
             and "__reeval" not in f.lower()]
    if cands:
        cands.sort()
        return os.path.join(metrics_dir, cands[-1])

    return None

def find_reeval_metrics(metrics_dir: str, trial: int, n: int, fixed_plan: str, sub: str | None):
    """
    Sucht Reeval CSV:
      - Modell 1: metrics_{trial}__Reeval_{fixed_plan}_Modell_1.{sub}.csv
      - Modell 2: metrics_{trial}__Reeval_{fixed_plan}_Modell_2.csv
    Fallback: metrics_{trial}__Reeval_*Modell_*.csv
    """
    if not os.path.isdir(metrics_dir):
        return None

    if n == 1 and sub is not None:
        strict = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_{fixed_plan}_Modell_1.{sub}.csv")
        if os.path.isfile(strict):
            return strict
        patt = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_*Modell_1.{sub}*.csv")
    else:
        strict = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_{fixed_plan}_Modell_{n}.csv")
        if os.path.isfile(strict):
            return strict
        patt = os.path.join(metrics_dir, f"metrics_{trial}__Reeval_*Modell_{n}*.csv")

    hits = sorted(glob(patt))
    return hits[0] if hits else None

def shorten_name_datenreduktion(study_name: str) -> str:
    """
    Format: <modell>_<reduction>_<seed>_<conf?>
    z.B. 1.2_CBIR_42_80  oder 2_RegMix_0
    """
    mdl  = get_model_str(study_name)
    red  = get_reduction_method(study_name)
    seed = get_seed_int(study_name)

    if red == "CBIR":
        conf = get_conf_int(study_name)
        conf_str = str(conf) if conf is not None else "?"
        return f"{mdl}_{red}_{seed}_{conf_str}"

    return f"{mdl}_{red}_{seed}"

def shade_by_reduction(ax, reds):
    for i, r in enumerate(reds):
        ax.axvspan(i - 0.5, i + 0.5, color=REDUCTION_COLOR_MAP.get(r, DEFAULT_BG), alpha=0.7, zorder=0)

def aggregate_per_reduction(values, reds):
    mean = {}
    std  = {}
    for r in REDUCTION_ORDER:
        vals = np.array([v for v, rr in zip(values, reds) if rr == r], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            mean[r] = np.nan
            std[r]  = np.nan
        else:
            mean[r] = float(vals.mean())
            std[r]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, std

# ============================================================
# Auswertung
# ============================================================
def auswertung_modell(n: int):
    plan_req  = PLAN_FIX.get(n)
    split_req = SPLIT_FIX.get(n)

    records = []

    for study in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, study)
        if not os.path.isdir(path):
            continue

        # Modellfilter
        if n == 1:
            if "Modell_1" not in study:
                continue
        elif n == 2:
            if "Modell_2" not in study:
                continue
        else:
            continue

        # Plan fix robust
        if plan_req and get_plan(study) != plan_req:
            continue

        # Split fix
        split = get_split_method(study)
        if split_req and split != split_req:
            continue

        # Reduktion
        red = get_reduction_method(study)
        if red not in REDUCTION_ORDER:
            continue

        # CBIR: nur CONF-Studies zulassen (SCR ignorieren)
        if red == "CBIR":
            if "_CONF_" not in study.upper():
                continue

        excel_path = pick_study_excel(path, study)
        if excel_path is None:
            continue

        try:
            df = pd.read_excel(excel_path, engine="openpyxl")
        except Exception:
            continue

        if not {"state", "value"}.issubset(df.columns):
            continue

        df_complete = df[df["state"] == "COMPLETE"]
        if df_complete.empty:
            continue

        best_idx = df_complete["value"].idxmin()
        best_trial = df_complete.loc[best_idx]

        if "number" in df_complete.columns:
            trial = int(best_trial["number"])
        elif "trial_id" in df_complete.columns:
            trial = int(best_trial["trial_id"])
        else:
            continue

        metrics_dir = os.path.join(path, "metrics")
        metrics_path = find_metrics_csv(metrics_dir, trial)
        if metrics_path is None:
            continue

        metrics = read_metrics(metrics_path)
        sub = get_submodel(study) if n == 1 else None
        reeval_path = find_reeval_metrics(metrics_dir, trial, n, plan_req, sub)
        metrics_own = read_metrics(reeval_path) if reeval_path else None

        mdl_str = get_model_str(study)
        seed = get_seed_int(study)
        conf = get_conf_int(study) if red == "CBIR" else None

        records.append({
            "label": shorten_name_datenreduktion(study),
            "red": red,
            "model_key": get_model_sortkey(mdl_str),
            "seed": seed,
            "conf": (conf if conf is not None else 999999),

            "rmse_train": metrics["train"]["rmse"],
            "rmse_val":   metrics["validation"]["rmse"],
            "rmse_test":  metrics["test"]["rmse"],

            "r2_train": metrics["train"]["r2"],
            "r2_val":   metrics["validation"]["r2"],
            "r2_test":  metrics["test"]["r2"],

            "rmse_test_own": (metrics_own["test"]["rmse"] if metrics_own else np.nan),
            "r2_test_own": (metrics_own["test"]["r2"] if metrics_own else np.nan)
        })

    if not records:
        print(f"Keine Studies für Modell {n} gefunden (Plan={plan_req}, Split={split_req}).")
        return

    # Sortierung: Reduktion -> Modell -> Seed -> CONF
    red_rank = {r: i for i, r in enumerate(REDUCTION_ORDER)}
    records.sort(key=lambda r: (
        red_rank.get(r["red"], 999),
        r["model_key"],
        r["seed"],
        r["conf"]
    ))

    labels = [r["label"] for r in records]
    reds   = [r["red"] for r in records]

    rmse_train = np.array([r["rmse_train"] for r in records], float)
    rmse_val   = np.array([r["rmse_val"]   for r in records], float)
    rmse_test  = np.array([r["rmse_test"]  for r in records], float)

    r2_train = np.array([r["r2_train"] for r in records], float)
    r2_val   = np.array([r["r2_val"]   for r in records], float)
    r2_test  = np.array([r["r2_test"]  for r in records], float)

    rmse_test_own = np.array([r["rmse_test_own"] for r in records], float)
    r2_test_own = np.array([r["r2_test_own"] for r in records], float)

    subfolder = f"Modell_{n}_Datenreduktion"

    # ================================
    # 1) RMSE Scatter
    # ================================
    fig, ax = plt.subplots(figsize=(12, 5))
    shade_by_reduction(ax, reds)

    x = np.arange(len(labels))
    ax.scatter(x, rmse_train, marker="o", label="Train", zorder=2, color="#95BB20")
    ax.scatter(x, rmse_val,   marker="o", label="Validation", zorder=2, color="#00354E")
    ax.scatter(x, rmse_test,  marker="o", label="Test global", zorder=2, color="#717E86")
    ax.scatter(x, rmse_test_own, marker="o", label="Test eigene Daten", zorder=3, color="#0098AD")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE aller besten Modelle (Modell {n}.*) – Datenreduktion (CBIR vs RegMix)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_{n}_Datenreduktion.png", subfolder)

    # ================================
    # 2) R² Scatter
    # ================================
    fig, ax = plt.subplots(figsize=(12, 5))
    shade_by_reduction(ax, reds)

    x = np.arange(len(labels))
    ax.scatter(x, r2_train, marker="o", label="Train", zorder=2, color="#95BB20")
    ax.scatter(x, r2_val,   marker="o", label="Validation", zorder=2, color="#00354E")
    ax.scatter(x, r2_test,  marker="o", label="Test global", zorder=2, color="#717E86")
    ax.scatter(x, r2_test_own, marker="o", label="Test eigene Daten", zorder=3, color="#0098AD")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center")
    ax.set_ylabel("R²")
    ax.set_title(f"R² aller besten Modelle (Modell {n}.*) – Datenreduktion (CBIR vs RegMix)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_{n}_Datenreduktion.png", subfolder)

    # ================================
    # 3) RMSE pro Reduktion – Balken (inkl. eigene Testdaten)
    # ================================
    mean_tr, std_tr = aggregate_per_reduction(rmse_train, reds)
    mean_v, std_v = aggregate_per_reduction(rmse_val, reds)
    mean_te, std_te = aggregate_per_reduction(rmse_test, reds)
    mean_own, std_own = aggregate_per_reduction(rmse_test_own, reds)

    x = np.arange(len(REDUCTION_ORDER))
    width = 0.20
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - 1.5 * width, [mean_tr[r] for r in REDUCTION_ORDER], width,
           yerr=[std_tr[r] for r in REDUCTION_ORDER], capsize=4,
           label="Train (global)", color="#95BB20", edgecolor="black")

    ax.bar(x - 0.5 * width, [mean_v[r] for r in REDUCTION_ORDER], width,
           yerr=[std_v[r] for r in REDUCTION_ORDER], capsize=4,
           label="Validation (global)", color="#00354E", edgecolor="black")

    ax.bar(x + 0.5 * width, [mean_te[r] for r in REDUCTION_ORDER], width,
           yerr=[std_te[r] for r in REDUCTION_ORDER], capsize=4,
           label="Test (global)", color="#717E86", edgecolor="black")

    ax.bar(x + 1.5 * width, [mean_own[r] for r in REDUCTION_ORDER], width,
           yerr=[std_own[r] for r in REDUCTION_ORDER], capsize=4,
           label="Test (eigene Daten)", color="#0098AD", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(REDUCTION_ORDER)
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE (Train/Val/Test global + Test eigene) pro Reduktionsmethode (Modell {n}.*)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f"RMSE_TrainValTest_Modell_{n}_by_reduction.png", subfolder)

    # ================================
    # 4) R² pro Reduktion – Balken (inkl. eigene Testdaten)
    # ================================
    mean_tr, std_tr = aggregate_per_reduction(r2_train, reds)
    mean_v, std_v = aggregate_per_reduction(r2_val, reds)
    mean_te, std_te = aggregate_per_reduction(r2_test, reds)
    mean_own, std_own = aggregate_per_reduction(r2_test_own, reds)

    x = np.arange(len(REDUCTION_ORDER))
    width = 0.20
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - 1.5 * width, [mean_tr[r] for r in REDUCTION_ORDER], width,
           yerr=[std_tr[r] for r in REDUCTION_ORDER], capsize=4,
           label="Train (global)", color="#95BB20", edgecolor="black")

    ax.bar(x - 0.5 * width, [mean_v[r] for r in REDUCTION_ORDER], width,
           yerr=[std_v[r] for r in REDUCTION_ORDER], capsize=4,
           label="Validation (global)", color="#00354E", edgecolor="black")

    ax.bar(x + 0.5 * width, [mean_te[r] for r in REDUCTION_ORDER], width,
           yerr=[std_te[r] for r in REDUCTION_ORDER], capsize=4,
           label="Test (global)", color="#717E86", edgecolor="black")

    ax.bar(x + 1.5 * width, [mean_own[r] for r in REDUCTION_ORDER], width,
           yerr=[std_own[r] for r in REDUCTION_ORDER], capsize=4,
           label="Test (eigene Daten)", color="#0098AD", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(REDUCTION_ORDER)
    ax.set_ylabel("R²")
    ax.set_title(f"R² (Train/Val/Test global + Test eigene) pro Reduktionsmethode (Modell {n}.*)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f"R2_TrainValTest_Modell_{n}_by_reduction.png", subfolder)

    print(f"[OK] Plots für Modell {n} gespeichert unter: {os.path.join(output_dir, subfolder)}")


# ============================================================
# Aufruf
# ============================================================
if __name__ == "__main__":
    auswertung_modell(1)
    auswertung_modell(2)