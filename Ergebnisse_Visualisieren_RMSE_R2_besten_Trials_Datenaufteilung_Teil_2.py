import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "Ergebnisse_Teil_2"
output_dir = "Ergebnisplots_Visualization_Metriken_Datenaufteilung"
os.makedirs(output_dir, exist_ok=True)

# Aufteilungsmethoden (Split) + Farben (Hintergrund)
SPLIT_ORDER = ["KS", "DUPLEX", "SPlit", "SPXY"]
SPLIT_COLOR_MAP = {
    "KS":     "#eef7ff",
    "DUPLEX": "#fff3e0",
    "SPlit":  "#e8f5e9",
    "SPXY":   "#fce4ec",
}

# ============================================================
# Helpers
# ============================================================
def save_plot(fig, filename, subfolder):
    save_path = os.path.join(output_dir, subfolder)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

def get_model_str(study_name: str) -> str:
    m = re.search(r"Modell_([\d.]+)", study_name)
    return m.group(1) if m else "?"

def get_model_sortkey(model_str: str):
        # "1.2" -> (1,2), "2" -> (2,0)
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
    m = re.search(r"seed_(\d+)", study_name)
    return int(m.group(1)) if m else 999999

def read_metrics(metrics_path):
    m = pd.read_csv(metrics_path)
    m.columns = [str(c).strip().lower() for c in m.columns]
    out = {
        "train": {"rmse": np.nan, "r2": np.nan},
        "validation": {"rmse": np.nan, "r2": np.nan},
        "test": {"rmse": np.nan, "r2": np.nan},
    }

    # Format 1: dataset/rmse/r2
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

    # Format 2: rmse_train/rmse_validation/...
    def get(col):
        return m[col].iloc[-1] if col in m.columns and not m[col].empty else np.nan

    for ds in out:
        out[ds]["rmse"] = get(f"rmse_{ds}")
        out[ds]["r2"]   = get(f"r2_{ds}")
    return out

def is_modell(study_name: str, n: int) -> bool:
    return re.search(rf"_Modell_{n}(?!\d)", study_name) is not None

def get_split_method(study_name: str) -> str:
    m = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name)
    return m.group(1) if m else "Other"

def get_plan(study_name: str) -> str:
    m = re.search(r"_(Halton|Sobol|Taguchi|LHS)_", study_name)
    return m.group(1) if m else "Other"

def shorten_name(study_name: str) -> str:
    # Beispiel: Study_..._Modell_1.1_KS_..._seed_0  -> 1.1_KS_0
    model = re.search(r"Modell_([\d.]+)", study_name)
    seed  = re.search(r"seed_(\d+)", study_name)
    split = get_split_method(study_name)
    mdl = model.group(1) if model else "?"
    sd  = seed.group(1) if seed else "?"
    return f"{mdl}_{split}_{sd}"

def pick_study_excel(study_path: str, study_name: str):
    preferred = os.path.join(study_path, f"{study_name}.xlsx")
    if os.path.isfile(preferred):
        return preferred

    xlsx_files = [f for f in os.listdir(study_path) if f.lower().endswith(".xlsx")]
    if not xlsx_files:
        return None

    xlsx_files.sort(key=lambda fn: os.path.getsize(os.path.join(study_path, fn)), reverse=True)
    return os.path.join(study_path, xlsx_files[0])

# ============================================================
# Auswertung
# ============================================================
def auswertung_modell(n: int):
    study_names = []
    split_names = []

    rmse_train, rmse_val, rmse_test = [], [], []
    r2_train,   r2_val,   r2_test   = [], [], []

    for study in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, study)
        if not os.path.isdir(path):
            continue
        if not is_modell(study, n):
            continue

        # Versuchsplan-Fixierung (wie von dir gewünscht)
        plan = get_plan(study)
        if n == 1 and plan != "Halton":
            continue
        if n == 2 and plan != "LHS":
            continue

        split = get_split_method(study)
        if split not in SPLIT_ORDER:
            continue

        excel_path = pick_study_excel(path, study)
        if excel_path is None:
            continue

        try:
            df = pd.read_excel(excel_path, engine="openpyxl")
        except Exception as e:
            print(f"[SKIP] Excel nicht lesbar: {excel_path} -> {type(e).__name__}: {e}")
            continue

        if not {"state", "value"}.issubset(df.columns):
            continue

        df_complete = df[df["state"] == "COMPLETE"]
        if df_complete.empty:
            continue

        # Trialnummer bestimmen
        best_idx = df_complete["value"].idxmin()
        best_trial = df_complete.loc[best_idx]

        if "number" in df_complete.columns:
            trial = int(best_trial["number"])
        elif "trial_id" in df_complete.columns:
            trial = int(best_trial["trial_id"])
        else:
            continue

        metrics_dir = os.path.join(path, "metrics")
        if not os.path.isdir(metrics_dir):
            continue

        metrics_path = os.path.join(metrics_dir, f"metrics_{trial}.csv")
        if not os.path.isfile(metrics_path):
            # fallback wie in Schritt 1
            csv_candidates = [f for f in os.listdir(metrics_dir)
                              if f.endswith(".csv") and str(trial) in f]
            if not csv_candidates:
                csv_candidates = [f for f in os.listdir(metrics_dir) if f.endswith(".csv")]
                if not csv_candidates:
                    continue
            metrics_path = os.path.join(metrics_dir, sorted(csv_candidates)[-1])

        metrics = read_metrics(metrics_path)

        study_names.append(shorten_name(study))
        split_names.append(split)

        rmse_train.append(metrics["train"]["rmse"])
        rmse_val.append(metrics["validation"]["rmse"])
        rmse_test.append(metrics["test"]["rmse"])

        r2_train.append(metrics["train"]["r2"])
        r2_val.append(metrics["validation"]["r2"])
        r2_test.append(metrics["test"]["r2"])

    if not study_names:
        print(f"Keine Daten für Modell {n}")
        return

    # ============================================================
    # ✅ Sortierung wie im "richtigen" Code:
    # Split-Block -> Modell -> Seed
    # ============================================================
    split_rank = {s: i for i, s in enumerate(SPLIT_ORDER)}

    model_keys = [get_model_sortkey(get_model_str(name)) for name in study_names]
    seed_keys  = [get_seed_int(name) for name in study_names]

    order = sorted(
        range(len(study_names)),
        key=lambda i: (split_rank.get(split_names[i], 999), model_keys[i], seed_keys[i])
    )

    study_names = [study_names[i] for i in order]
    split_names = [split_names[i] for i in order]

    rmse_train = [rmse_train[i] for i in order]
    rmse_val   = [rmse_val[i]   for i in order]
    rmse_test  = [rmse_test[i]  for i in order]

    r2_train = [r2_train[i] for i in order]
    r2_val   = [r2_val[i]   for i in order]
    r2_test  = [r2_test[i]  for i in order]

    subfolder = f"Modell_{n}"

    def aggregate_per_split(values_dict):
        splits, means, stds = [], [], []
        for sp in SPLIT_ORDER:
            if sp not in values_dict:
                continue
            vals = np.array(values_dict[sp], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            splits.append(sp)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        return splits, means, stds

    # ================================
    # 1) RMSE Train/Val/Test – Scatter
    # ================================
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sp in enumerate(split_names):
        ax.axvspan(i - 0.5, i + 0.5, color=SPLIT_COLOR_MAP.get(sp, "#ffffff"), alpha=0.7, zorder=0)

    ax.scatter(study_names, rmse_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, rmse_val,   marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, rmse_test,  marker="o", label="Test", zorder=1, color="#717E86")

    ax.set_ylabel("RMSE")
    ax.legend()
    ax.set_ylim(top=16)
    ax.set_title(f"RMSE aller besten Modelle (Modell {n}.*) – Aufteilungsmethoden")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_{n}.png", subfolder)

    # ================================
    # 2) R² Train/Val/Test – Scatter
    # ================================
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sp in enumerate(split_names):
        ax.axvspan(i - 0.5, i + 0.5, color=SPLIT_COLOR_MAP.get(sp, "#ffffff"), alpha=0.7, zorder=0)

    ax.scatter(study_names, r2_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, r2_val,   marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, r2_test,  marker="o", label="Test", zorder=1, color="#717E86")

    ax.set_ylabel("R²")
    ax.legend()
    ax.set_title(f"R² aller besten Modelle (Modell {n}.*) – Aufteilungsmethoden")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_{n}.png", subfolder)

    # ================================
    # 3) RMSE (Train/Val/Test) pro Split – gruppierte Balken
    # ================================
    split_to_rmse_train, split_to_rmse_val, split_to_rmse_test = {}, {}, {}
    for sp, rm_tr, rm_v, rm_te in zip(split_names, rmse_train, rmse_val, rmse_test):
        if sp not in SPLIT_COLOR_MAP:
            continue
        if not pd.isna(rm_tr): split_to_rmse_train.setdefault(sp, []).append(rm_tr)
        if not pd.isna(rm_v):  split_to_rmse_val.setdefault(sp, []).append(rm_v)
        if not pd.isna(rm_te): split_to_rmse_test.setdefault(sp, []).append(rm_te)

    splits, mean_tr, std_tr = aggregate_per_split(split_to_rmse_train)
    _,      mean_v,  std_v  = aggregate_per_split(split_to_rmse_val)
    _,      mean_te, std_te = aggregate_per_split(split_to_rmse_test)

    if splits:
        x = np.arange(len(splits))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr,  width, yerr=std_tr,  capsize=4, label="Train",      color="#95BB20", edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,   capsize=4, label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te,  capsize=4, label="Test",       color="#717E86", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel("RMSE")
        ax.set_ylim(top=15)
        ax.set_title(f"RMSE (Train/Val/Test) pro Aufteilungsmethode (Modell {n}.*)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_TrainValTest_Modell_{n}_by_split.png", subfolder)

    # ================================
    # 4) R² (Train/Val/Test) pro Split – gruppierte Balken
    # ================================
    split_to_r2_train, split_to_r2_val, split_to_r2_test = {}, {}, {}
    for sp, r2_tr, r2_v, r2_te in zip(split_names, r2_train, r2_val, r2_test):
        if sp not in SPLIT_COLOR_MAP:
            continue
        if not pd.isna(r2_tr): split_to_r2_train.setdefault(sp, []).append(r2_tr)
        if not pd.isna(r2_v):  split_to_r2_val.setdefault(sp, []).append(r2_v)
        if not pd.isna(r2_te): split_to_r2_test.setdefault(sp, []).append(r2_te)

    splits, mean_tr, std_tr = aggregate_per_split(split_to_r2_train)
    _,      mean_v,  std_v  = aggregate_per_split(split_to_r2_val)
    _,      mean_te, std_te = aggregate_per_split(split_to_r2_test)

    if splits:
        x = np.arange(len(splits))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr,  width, yerr=std_tr,  capsize=4, label="Train",      color="#95BB20", edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,   capsize=4, label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te,  capsize=4, label="Test",       color="#717E86", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel("R²")
        ax.set_title(f"R² (Train/Val/Test) pro Aufteilungsmethode (Modell {n}.*)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_TrainValTest_Modell_{n}_by_split.png", subfolder)

    # ================================
    # 5) RMSE(Test) Scatter – nach RMSE sortiert
    # ================================
    order_rmse = np.argsort(np.array(rmse_test, dtype=float))
    sorted_names   = [study_names[i] for i in order_rmse]
    sorted_rmse    = [rmse_test[i]   for i in order_rmse]
    sorted_splits  = [split_names[i] for i in order_rmse]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sp in enumerate(sorted_splits):
        ax.axvspan(i - 0.5, i + 0.5, color=SPLIT_COLOR_MAP.get(sp, "#ffffff"), alpha=0.7, zorder=0)

    ax.scatter(sorted_names, sorted_rmse, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("RMSE Test")
    ax.set_title(f"RMSE (Test) über alle besten Modelle (Modell {n}.*) – Aufteilungsmethoden")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_Test_Modell_{n}.png", subfolder)

    # ================================
    # 6) R²(Test) Scatter – nach R² sortiert (absteigend)
    # ================================
    r2_arr = np.array([np.nan if pd.isna(x) else float(x) for x in r2_test], dtype=float)
    order_r2 = np.argsort(-r2_arr)  # absteigend
    sorted_names   = [study_names[i] for i in order_r2]
    sorted_r2      = [r2_test[i]     for i in order_r2]
    sorted_splits  = [split_names[i] for i in order_r2]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sp in enumerate(sorted_splits):
        ax.axvspan(i - 0.5, i + 0.5, color=SPLIT_COLOR_MAP.get(sp, "#ffffff"), alpha=0.7, zorder=0)

    ax.scatter(sorted_names, sorted_r2, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("R² Test")
    ax.set_title(f"R² (Test) über alle besten Modelle (Modell {n}.*) – Aufteilungsmethoden")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_Test_Modell_{n}.png", subfolder)

    # ================================
    # 7) RMSE(Test) pro Split – Balken
    # ================================
    split_to_rmse = {}
    for rm, sp in zip(rmse_test, split_names):
        if pd.isna(rm):
            continue
        if sp not in SPLIT_COLOR_MAP:
            continue
        split_to_rmse.setdefault(sp, []).append(rm)

    splits, means, stds = aggregate_per_split(split_to_rmse)
    if splits:
        x = np.arange(len(splits))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [SPLIT_COLOR_MAP.get(sp, "#cccccc") for sp in splits]

        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel("RMSE Test")
        ax.set_title(f"RMSE (Test) pro Aufteilungsmethode (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_Test_Modell_{n}_by_split.png", subfolder)

    # ================================
    # 8) R²(Test) pro Split – Balken
    # ================================
    split_to_r2 = {}
    for r2v, sp in zip(r2_test, split_names):
        if pd.isna(r2v):
            continue
        if sp not in SPLIT_COLOR_MAP:
            continue
        split_to_r2.setdefault(sp, []).append(r2v)

    splits, means, stds = aggregate_per_split(split_to_r2)
    if splits:
        x = np.arange(len(splits))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [SPLIT_COLOR_MAP.get(sp, "#cccccc") for sp in splits]

        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel("R² Test")
        ax.set_title(f"R² (Test) pro Aufteilungsmethode (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_Test_Modell_{n}_by_split.png", subfolder)

    print(f"Plots für Modell {n} gespeichert unter: {os.path.join(output_dir, subfolder)}")

# ============================================================
# Aufruf
# ============================================================
auswertung_modell(1)
auswertung_modell(2)