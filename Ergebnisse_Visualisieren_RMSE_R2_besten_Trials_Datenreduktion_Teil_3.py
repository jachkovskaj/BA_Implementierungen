import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Konfiguration
# ============================================================
base_dir = "Ergebnisse_Teil_3"
output_dir = "Ergebnisplots_Visualization_Metriken_Datenreduktion"
os.makedirs(output_dir, exist_ok=True)

REDUCTION_ORDER = ["CBIR", "RegMix"]
REDUCTION_COLOR_MAP = {
    "CBIR":   "#eef7ff",  # blau
    "RegMix": "#fff3e0",  # gelb
}

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

def get_plan(study_name: str) -> str:
    """
    Wichtig: In deinen Teil-3 Ordnernamen können mehrere Plan-Tokens vorkommen
    (z.B. "..._Halton_Modell_2_LHS_..."). Dann soll für Modell 2 LHS gelten.
    => Wir nehmen den *letzten* gefundenen Plan-Token.
    """
    matches = re.findall(r"_(Halton|Sobol|Taguchi|LHS)_", study_name)
    return matches[-1] if matches else "Other"

def get_split_method(study_name: str) -> str:
    m = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name)
    return m.group(1) if m else "Other"

def get_reduction_method(study_name: str) -> str:
    m = re.search(r"_(CBIR|RegMix)_", study_name, re.IGNORECASE)
    if not m:
        return "Other"
    red = m.group(1).lower()
    return "CBIR" if red == "cbir" else ("RegMix" if red == "regmix" else "Other")

def get_model_label_from_dirname(study_name: str) -> str:
    m = re.search(r"Modell_([\d.]+)", study_name)
    return m.group(1) if m else "?"

def get_seed_from_dirname(study_name: str) -> int:
    m = re.search(r"seed_(\d+)", study_name)
    return int(m.group(1)) if m else 999999

def shorten_name(study_name: str) -> str:
    """
    Beispiel:
      Study_..._Modell_1.2_..._CBIR_..._CONF_85_..._seed_42
      -> 1.2_CBIR_42_85
    """
    model = re.search(r"Modell_([\d.]+)", study_name)
    seed  = re.search(r"seed_(\d+)", study_name)
    red   = get_reduction_method(study_name)
    conf  = re.search(r"_CONF_(\d+)", study_name)

    mdl = model.group(1) if model else "?"
    sd  = seed.group(1) if seed else "?"
    cf  = conf.group(1) if conf else None

    if red == "CBIR" and cf is not None:
        return f"{mdl}_{red}_{sd}_{cf}"

    return f"{mdl}_{red}_{sd}"

def pick_study_excel(study_path: str, study_name: str):
    preferred = os.path.join(study_path, f"{study_name}.xlsx")
    if os.path.isfile(preferred):
        return preferred

    xlsx_files = [f for f in os.listdir(study_path) if f.lower().endswith(".xlsx")]
    if not xlsx_files:
        return None

    xlsx_files.sort(key=lambda fn: os.path.getsize(os.path.join(study_path, fn)), reverse=True)
    return os.path.join(study_path, xlsx_files[0])

def find_metrics_csv(metrics_dir: str, trial: int) -> str | None:
    """
    ✅ Sucht NUR in /metrics nach metrics_*.csv (keine Reeval-Dateien).
    Bevorzugt exakt metrics_{trial}.csv, sonst tolerant metrics_*{trial}*.csv.
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

    # fallback: irgendeine metrics_*.csv (aber keine __Reeval)
    cands = [f for f in os.listdir(metrics_dir)
             if f.lower().endswith(".csv")
             and f.lower().startswith("metrics_")
             and "__reeval" not in f.lower()]
    if cands:
        cands.sort()
        return os.path.join(metrics_dir, cands[-1])

    return None

def draw_background_blocks(ax, group_names, color_map, alpha=0.7):
    """
    Zeichnet Hintergrund *blockweise* anhand von zusammenhängenden gleichen group_names.
    group_names ist eine Liste (z.B. ["CBIR","CBIR","CBIR","RegMix",...]) in Plot-Reihenfolge.
    """
    if not group_names:
        return
    start = 0
    for i in range(1, len(group_names)):
        if group_names[i] != group_names[i-1]:
            g = group_names[start]
            ax.axvspan(start - 0.5, (i-1) + 0.5, color=color_map.get(g, "#ffffff"), alpha=alpha, zorder=0)
            start = i
    g = group_names[start]
    ax.axvspan(start - 0.5, (len(group_names)-1) + 0.5, color=color_map.get(g, "#ffffff"), alpha=alpha, zorder=0)

# ============================================================
# Auswertung
# ============================================================
def auswertung_modell(n: int):
    plan_req  = PLAN_FIX.get(n)
    split_req = SPLIT_FIX.get(n)

    labels_for_plot = []   # z.B. "1.1_CBIR_0"
    red_names = []         # "CBIR"/"RegMix"
    model_labels = []      # "1.1" / "2"
    seed_vals = []         # int

    rmse_train, rmse_val, rmse_test = [], [], []
    r2_train,   r2_val,   r2_test   = [], [], []

    for study in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, study)
        if not os.path.isdir(path):
            continue

        # Modellfilter
        if n == 1 and "Modell_1" not in study:
            continue
        if n == 2 and "Modell_2" not in study:
            continue
        if n not in (1, 2):
            continue

        # Plan fix (letzter Token!)
        plan = get_plan(study)
        if plan_req and plan != plan_req:
            continue

        # Split fix
        split = get_split_method(study)
        if split_req and split != split_req:
            continue

        # Reduktion
        red = get_reduction_method(study)
        if red not in REDUCTION_ORDER:
            continue

        if red == "CBIR" and "_CONF_" not in study:
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

        labels_for_plot.append(shorten_name(study))
        red_names.append(red)
        model_labels.append(get_model_label_from_dirname(study))
        seed_vals.append(get_seed_from_dirname(study))

        rmse_train.append(metrics["train"]["rmse"])
        rmse_val.append(metrics["validation"]["rmse"])
        rmse_test.append(metrics["test"]["rmse"])

        r2_train.append(metrics["train"]["r2"])
        r2_val.append(metrics["validation"]["r2"])
        r2_test.append(metrics["test"]["r2"])

    if not labels_for_plot:
        print(f"Keine Daten für Modell {n} (Plan={plan_req}, Split={split_req})")
        return

    # ============================================================
    # Sortierung: Reduktion -> Modell -> Seed
    # ============================================================
    red_rank = {r: i for i, r in enumerate(REDUCTION_ORDER)}
    model_keys = [get_model_sortkey(m) for m in model_labels]

    order = sorted(
        range(len(labels_for_plot)),
        key=lambda i: (red_rank.get(red_names[i], 999), model_keys[i], seed_vals[i])
    )

    labels_for_plot = [labels_for_plot[i] for i in order]
    red_names       = [red_names[i]       for i in order]

    rmse_train = [rmse_train[i] for i in order]
    rmse_val   = [rmse_val[i]   for i in order]
    rmse_test  = [rmse_test[i]  for i in order]

    r2_train = [r2_train[i] for i in order]
    r2_val   = [r2_val[i]   for i in order]
    r2_test  = [r2_test[i]  for i in order]

    subfolder = f"Modell_{n}"

    def aggregate_per_reduction(values_dict):
        reds, means, stds = [], [], []
        for r in REDUCTION_ORDER:
            if r not in values_dict:
                continue
            vals = np.array(values_dict[r], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            reds.append(r)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        return reds, means, stds

    # feste x-Positionen (=> kein “links verschoben” mehr)
    x = np.arange(len(labels_for_plot))

    # ================================
    # 1) RMSE Train/Val/Test – Scatter (CBIR/RegMix Hintergrund blockweise)
    # ================================
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_background_blocks(ax, red_names, REDUCTION_COLOR_MAP, alpha=0.7)

    ax.scatter(x, rmse_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(x, rmse_val,   marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(x, rmse_test,  marker="o", label="Test", zorder=1, color="#717E86")

    ax.set_ylabel("RMSE")
    ax.legend()
    ax.set_title(f"RMSE aller besten Modelle (Modell {n}.*) – Datenreduktion (CBIR vs RegMix)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_for_plot, rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_{n}_by_reduction.png", subfolder)

    # ================================
    # 2) R² Train/Val/Test – Scatter
    # ================================
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_background_blocks(ax, red_names, REDUCTION_COLOR_MAP, alpha=0.7)

    ax.scatter(x, r2_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(x, r2_val,   marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(x, r2_test,  marker="o", label="Test", zorder=1, color="#717E86")

    ax.set_ylabel("R²")
    ax.legend()
    ax.set_title(f"R² aller besten Modelle (Modell {n}.*) – Datenreduktion (CBIR vs RegMix)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_for_plot, rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_{n}_by_reduction.png", subfolder)

    # ================================
    # 3) RMSE (Train/Val/Test) pro Reduktion – gruppierte Balken
    # ================================
    red_to_rmse_train, red_to_rmse_val, red_to_rmse_test = {}, {}, {}
    for r, rm_tr, rm_v, rm_te in zip(red_names, rmse_train, rmse_val, rmse_test):
        if r not in REDUCTION_ORDER:
            continue
        if not pd.isna(rm_tr): red_to_rmse_train.setdefault(r, []).append(rm_tr)
        if not pd.isna(rm_v):  red_to_rmse_val.setdefault(r, []).append(rm_v)
        if not pd.isna(rm_te): red_to_rmse_test.setdefault(r, []).append(rm_te)

    reds, mean_tr, std_tr = aggregate_per_reduction(red_to_rmse_train)
    _,    mean_v,  std_v  = aggregate_per_reduction(red_to_rmse_val)
    _,    mean_te, std_te = aggregate_per_reduction(red_to_rmse_test)

    if reds:
        xb = np.arange(len(reds))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(xb - width, mean_tr,  width, yerr=std_tr,  capsize=4, label="Train",      color="#95BB20", edgecolor="black")
        ax.bar(xb,         mean_v,  width, yerr=std_v,   capsize=4, label="Validation", color="#00354E", edgecolor="black")
        ax.bar(xb + width, mean_te, width, yerr=std_te,  capsize=4, label="Test",       color="#717E86", edgecolor="black")

        ax.set_xticks(xb)
        ax.set_xticklabels(reds)
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE (Train/Val/Test) pro Reduktionsmethode (Modell {n}.*)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_TrainValTest_Modell_{n}_by_reduction.png", subfolder)

    # ================================
    # 4) R² (Train/Val/Test) pro Reduktion – gruppierte Balken
    # ================================
    red_to_r2_train, red_to_r2_val, red_to_r2_test = {}, {}, {}
    for r, r2_tr, r2_v, r2_te in zip(red_names, r2_train, r2_val, r2_test):
        if r not in REDUCTION_ORDER:
            continue
        if not pd.isna(r2_tr): red_to_r2_train.setdefault(r, []).append(r2_tr)
        if not pd.isna(r2_v):  red_to_r2_val.setdefault(r, []).append(r2_v)
        if not pd.isna(r2_te): red_to_r2_test.setdefault(r, []).append(r2_te)

    reds, mean_tr, std_tr = aggregate_per_reduction(red_to_r2_train)
    _,    mean_v,  std_v  = aggregate_per_reduction(red_to_r2_val)
    _,    mean_te, std_te = aggregate_per_reduction(red_to_r2_test)

    if reds:
        xb = np.arange(len(reds))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(xb - width, mean_tr,  width, yerr=std_tr,  capsize=4, label="Train",      color="#95BB20", edgecolor="black")
        ax.bar(xb,         mean_v,  width, yerr=std_v,   capsize=4, label="Validation", color="#00354E", edgecolor="black")
        ax.bar(xb + width, mean_te, width, yerr=std_te,  capsize=4, label="Test",       color="#717E86", edgecolor="black")

        ax.set_xticks(xb)
        ax.set_xticklabels(reds)
        ax.set_ylabel("R²")
        ax.set_title(f"R² (Train/Val/Test) pro Reduktionsmethode (Modell {n}.*)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_TrainValTest_Modell_{n}_by_reduction.png", subfolder)

    # ================================
    # 5) RMSE(Test) Scatter – nach RMSE sortiert
    # ================================
    order_rmse = np.argsort(np.array(rmse_test, dtype=float))
    sorted_labels = [labels_for_plot[i] for i in order_rmse]
    sorted_rmse   = [rmse_test[i]       for i in order_rmse]
    sorted_reds   = [red_names[i]       for i in order_rmse]
    xs = np.arange(len(sorted_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    draw_background_blocks(ax, sorted_reds, REDUCTION_COLOR_MAP, alpha=0.7)

    ax.scatter(xs, sorted_rmse, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("RMSE Test")
    ax.set_title(f"RMSE (Test) über alle besten Modelle (Modell {n}.*) – Datenreduktion")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xticklabels(sorted_labels, rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_Test_Modell_{n}_by_reduction.png", subfolder)

    # ================================
    # 6) R²(Test) Scatter – nach R² sortiert (absteigend)
    # ================================
    r2_arr = np.array([np.nan if pd.isna(v) else float(v) for v in r2_test], dtype=float)
    order_r2 = np.argsort(-r2_arr)

    sorted_labels = [labels_for_plot[i] for i in order_r2]
    sorted_r2     = [r2_test[i]         for i in order_r2]
    sorted_reds   = [red_names[i]       for i in order_r2]
    xs = np.arange(len(sorted_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    draw_background_blocks(ax, sorted_reds, REDUCTION_COLOR_MAP, alpha=0.7)

    ax.scatter(xs, sorted_r2, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("R² Test")
    ax.set_title(f"R² (Test) über alle besten Modelle (Modell {n}.*) – Datenreduktion")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xticklabels(sorted_labels, rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_Test_Modell_{n}_by_reduction.png", subfolder)

    # ================================
    # 7) RMSE(Test) pro Reduktion – Balken
    # ================================
    red_to_rmse = {}
    for rm, r in zip(rmse_test, red_names):
        if pd.isna(rm):
            continue
        if r not in REDUCTION_ORDER:
            continue
        red_to_rmse.setdefault(r, []).append(rm)

    reds, means, stds = aggregate_per_reduction(red_to_rmse)
    if reds:
        xb = np.arange(len(reds))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [REDUCTION_COLOR_MAP.get(r, "#cccccc") for r in reds]

        ax.bar(xb, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax.set_xticks(xb)
        ax.set_xticklabels(reds)
        ax.set_ylabel("RMSE Test")
        ax.set_title(f"RMSE (Test) pro Reduktionsmethode (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_Test_Modell_{n}_bar_by_reduction.png", subfolder)

    # ================================
    # 8) R²(Test) pro Reduktion – Balken
    # ================================
    red_to_r2 = {}
    for r2v, r in zip(r2_test, red_names):
        if pd.isna(r2v):
            continue
        if r not in REDUCTION_ORDER:
            continue
        red_to_r2.setdefault(r, []).append(r2v)

    reds, means, stds = aggregate_per_reduction(red_to_r2)
    if reds:
        xb = np.arange(len(reds))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [REDUCTION_COLOR_MAP.get(r, "#cccccc") for r in reds]

        ax.bar(xb, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax.set_xticks(xb)
        ax.set_xticklabels(reds)
        ax.set_ylabel("R² Test")
        ax.set_title(f"R² (Test) pro Reduktionsmethode (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_Test_Modell_{n}_bar_by_reduction.png", subfolder)

    print(f"[OK] Plots für Modell {n} gespeichert unter: {os.path.join(output_dir, subfolder)}")

# ============================================================
# Aufruf
# ============================================================
auswertung_modell(1)
auswertung_modell(2)