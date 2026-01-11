import os
import re
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Konfiguration
# ============================================================
base_dir = "Ergebnisse_Teil_3"
output_dir = "Ergebnisplots_Visualization_Metriken_kombiniert_Datenreduktion"
os.makedirs(output_dir, exist_ok=True)

# Pro Reduktionsmethode EIN Plotpaket, das Modell 1 und 2 gemeinsam enthält
REDUCTION_ORDER = ["CBIR", "RegMix"]

# Punkt-/Balkenfarben (wie dein Original)
C_TRAIN = "#95BB20"   # hellgrün
C_VAL   = "#00354E"   # dunkelblau
C_GLOB  = "#717E86"   # grau
C_OWN   = "#0098AD"   # hell türkis

# Hintergrundfarben nach Modell (wie gewünscht)
BG_MODEL = {
    1: "#eef7ff",   # blau
    2: "#fff3e0",   # gelb
}

# Fixierungen
PLAN_FIX  = {1: "Halton", 2: "LHS"}
SPLIT_FIX = {1: "SPXY",   2: "KS"}

MODEL_ORDER = [1, 2]
MODEL_LABEL = {1: "Modell 1", 2: "Modell 2"}


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
                out[ds]["r2"] = float(g.loc[ds, "r2"])
        return out

    def get(col):
        return m[col].iloc[-1] if col in m.columns and not m[col].empty else np.nan

    for ds in out:
        out[ds]["rmse"] = get(f"rmse_{ds}")
        out[ds]["r2"] = get(f"r2_{ds}")
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
    """Sucht NUR in /metrics nach metrics_*.csv (keine Reeval)."""
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


def find_reeval_metrics(metrics_dir: str, trial: int, n: int, fixed_plan: str, sub: Optional[str]):
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
    mdl = get_model_str(study_name)
    red = get_reduction_method(study_name)
    seed = get_seed_int(study_name)

    if red == "CBIR":
        conf = get_conf_int(study_name)
        conf_str = str(conf) if conf is not None else "?"
        return f"{mdl}_{red}_{seed}_{conf_str}"

    return f"{mdl}_{red}_{seed}"


# ============================================================
# Datensammlung
# ============================================================
def collect_records_for_model(n: int):
    plan_req = PLAN_FIX.get(n)
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

        # Plan fix
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

        # CBIR: nur CONF-Studies zulassen
        if red == "CBIR" and "_CONF_" not in study.upper():
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
            "model": n,
            "red": red,
            "label": shorten_name_datenreduktion(study),
            "model_key": get_model_sortkey(mdl_str),
            "seed": seed,
            "conf": (conf if conf is not None else 999999),

            "rmse_train": metrics["train"]["rmse"],
            "rmse_val": metrics["validation"]["rmse"],
            "rmse_test": metrics["test"]["rmse"],

            "r2_train": metrics["train"]["r2"],
            "r2_val": metrics["validation"]["r2"],
            "r2_test": metrics["test"]["r2"],

            "rmse_test_own": (metrics_own["test"]["rmse"] if metrics_own else np.nan),
            "r2_test_own": (metrics_own["test"]["r2"] if metrics_own else np.nan),
        })

    return records


def sort_records_internally(recs):
    # stabile Reihenfolge
    recs.sort(key=lambda r: (r["model"], r["model_key"], r["seed"], r["conf"]))
    return recs


# ============================================================
# Plotting
# ============================================================
def plot_scatter_for_reduction(reduction: str, all_records: list, subfolder: str):
    """
    Plot 1+2 (Scatter):
    - Hintergrund: Modell 1 blau, Modell 2 gelb (pro Label-Block)
    - Punktefarben: Train/Val/Test/Test-own wie im Original
    """
    recs = [r for r in all_records if r["red"] == reduction]
    if not recs:
        return

    labels_all = sorted({r["label"] for r in recs})
    x = np.arange(len(labels_all))

    # Modell pro Label bestimmen (für Hintergrund)
    label_to_model = {}
    for r in recs:
        label_to_model.setdefault(r["label"], r["model"])

    def values_for(model, key):
        d = {r["label"]: r[key] for r in recs if r["model"] == model}
        return np.array([d.get(lbl, np.nan) for lbl in labels_all], dtype=float)

    # ---------------- RMSE Scatter ----------------
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, lbl in enumerate(labels_all):
        m = label_to_model.get(lbl, None)
        ax.axvspan(i - 0.5, i + 0.5, color=BG_MODEL.get(m, "#ffffff"), alpha=0.85, zorder=0)

    for model in MODEL_ORDER:
        dx = -0.12 if model == 1 else 0.12

        ax.scatter(x + dx, values_for(model, "rmse_train"), marker="o",
                   label=f"Train ({MODEL_LABEL[model]})", zorder=3, color=C_TRAIN)
        ax.scatter(x + dx, values_for(model, "rmse_val"), marker="o",
                   label=f"Validation ({MODEL_LABEL[model]})", zorder=3, color=C_VAL)
        ax.scatter(x + dx, values_for(model, "rmse_test"), marker="o",
                   label=f"Test global ({MODEL_LABEL[model]})", zorder=3, color=C_GLOB)
        ax.scatter(x + dx, values_for(model, "rmse_test_own"), marker="o",
                   label=f"Test eigene Daten ({MODEL_LABEL[model]})", zorder=4, color=C_OWN)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_all, rotation=90, ha="center")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE – Datenreduktion: {reduction} (Modell 1 vs Modell 2)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    save_plot(fig, f"RMSE_{reduction}_Modell1_vs_Modell2.png", subfolder)

    # ---------------- R² Scatter ----------------
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, lbl in enumerate(labels_all):
        m = label_to_model.get(lbl, None)
        ax.axvspan(i - 0.5, i + 0.5, color=BG_MODEL.get(m, "#ffffff"), alpha=0.85, zorder=0)

    for model in MODEL_ORDER:
        dx = -0.12 if model == 1 else 0.12

        ax.scatter(x + dx, values_for(model, "r2_train"), marker="o",
                   label=f"Train ({MODEL_LABEL[model]})", zorder=3, color=C_TRAIN)
        ax.scatter(x + dx, values_for(model, "r2_val"), marker="o",
                   label=f"Validation ({MODEL_LABEL[model]})", zorder=3, color=C_VAL)
        ax.scatter(x + dx, values_for(model, "r2_test"), marker="o",
                   label=f"Test global ({MODEL_LABEL[model]})", zorder=3, color=C_GLOB)
        ax.scatter(x + dx, values_for(model, "r2_test_own"), marker="o",
                   label=f"Test eigene Daten ({MODEL_LABEL[model]})", zorder=4, color=C_OWN)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_all, rotation=90, ha="center")
    ax.set_ylabel("R²")
    ax.set_ylim(top=1.6)
    ax.set_title(f"R² – Datenreduktion: {reduction} (Modell 1 vs Modell 2)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    save_plot(fig, f"R2_{reduction}_Modell1_vs_Modell2.png", subfolder)


def plot_bars_for_reduction(reduction: str, all_records: list, subfolder: str):
    """
    Plot 3+4 (Balken):
    - x-Achse: Modell 1, Modell 2 (nach Modellen sortiert)
    - pro Modell: 4 Balken (Train/Val/Test global/Test own) in den 4 gegebenen Farben
    - Hintergrund: Modell 1 blau, Modell 2 gelb (pro Modell-Gruppe)
    - Fehlerbalken: Std über Seeds/Studien
    """
    recs = [r for r in all_records if r["red"] == reduction]
    if not recs:
        return

    def agg(model, key):
        vals = np.array([r[key] for r in recs if r["model"] == model], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        return mean, std

    ds_labels = ["Train (global)", "Validation (global)", "Test (global)", "Test (eigene Daten)"]
    keys_rmse = ["rmse_train", "rmse_val", "rmse_test", "rmse_test_own"]
    keys_r2   = ["r2_train",   "r2_val",   "r2_test",   "r2_test_own"]
    ds_colors = [C_TRAIN, C_VAL, C_GLOB, C_OWN]

    xg = np.arange(len(MODEL_ORDER))  # [0, 1]
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    # ---------------- RMSE Bars ----------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for j, (lab, key, col) in enumerate(zip(ds_labels, keys_rmse, ds_colors)):
        means = [agg(m, key)[0] for m in MODEL_ORDER]
        stds  = [agg(m, key)[1] for m in MODEL_ORDER]
        ax.bar(xg + offsets[j], means, width,
               yerr=stds, capsize=4, label=lab,
               color=col, edgecolor="black", zorder=2)

    ax.set_xticks(xg)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE – {reduction}: Vergleich Modell 1 vs Modell 2 (Mittelwert ± Std)")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    save_plot(fig, f"RMSE__Train_Test_Val_{reduction}_Modell1_vs_Modell2.png", subfolder)

    # ---------------- R² Bars ----------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for j, (lab, key, col) in enumerate(zip(ds_labels, keys_r2, ds_colors)):
        means = [agg(m, key)[0] for m in MODEL_ORDER]
        stds  = [agg(m, key)[1] for m in MODEL_ORDER]
        ax.bar(xg + offsets[j], means, width,
               yerr=stds, capsize=4, label=lab,
               color=col, edgecolor="black", zorder=2)

    ax.set_xticks(xg)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_ylabel("R²")
    ax.set_title(f"R² – {reduction}: Vergleich Modell 1 vs Modell 2 (Mittelwert ± Std)")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    save_plot(fig, f"R2_Train_Test_Val_{reduction}_Modell1_vs_Modell2.png", subfolder)


# ============================================================
# Orchestrierung
# ============================================================
def auswertung_datenreduktion_beide_modelle():
    all_records = []
    all_records.extend(collect_records_for_model(1))
    all_records.extend(collect_records_for_model(2))

    if not all_records:
        print("Keine passenden Studies für Datenreduktion gefunden.")
        return

    all_records = sort_records_internally(all_records)
    subfolder = "Datenreduktion_Modell1_vs_Modell2"

    for red in REDUCTION_ORDER:
        plot_scatter_for_reduction(red, all_records, subfolder=subfolder)
        plot_bars_for_reduction(red, all_records, subfolder=subfolder)

    print(f"[OK] Plots gespeichert unter: {os.path.join(output_dir, subfolder)}")


# ============================================================
# Aufruf
# ============================================================
if __name__ == "__main__":
    auswertung_datenreduktion_beide_modelle()