import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "Ergebnisse_Teil_2"
output_dir = "Ergebnisplots_Visualization_Metriken_Aufteilungsmethode_eigene_Daten"
os.makedirs(output_dir, exist_ok=True)


def save_plot(fig, filename, subfolder):
    """Hilfsfunktion zum Speichern jedes Plots."""
    save_path = os.path.join(output_dir, subfolder)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_metrics(metrics_path):
    m = pd.read_csv(metrics_path)
    m.columns = [str(c).strip().lower() for c in m.columns]
    out = {
        'train': {'rmse': np.nan, 'r2': np.nan},
        'validation': {'rmse': np.nan, 'r2': np.nan},
        'test': {'rmse': np.nan, 'r2': np.nan}
    }

    if {'dataset', 'rmse', 'r2'}.issubset(m.columns):
        norm = {
            'train': 'train', 'training': 'train',
            'valid': 'validation', 'validation': 'validation', 'val': 'validation',
            'test': 'test', 'testing': 'test'
        }
        m['dataset'] = m['dataset'].astype(str).str.strip().str.lower().map(norm)
        g = (m.dropna(subset=['dataset'])
               .groupby('dataset', sort=False)[['rmse', 'r2']]
               .agg('last'))
        for ds in out:
            if ds in g.index:
                out[ds]['rmse'] = float(g.loc[ds, 'rmse'])
                out[ds]['r2']   = float(g.loc[ds, 'r2'])
        return out

    def get(col):
        return m[col].iloc[-1] if col in m.columns and not m[col].empty else np.nan

    for ds in out:
        out[ds]['rmse'] = get(f'rmse_{ds}')
        out[ds]['r2']   = get(f'r2_{ds}')
    return out


# Helfer: Ordnername gehört zu Modell N?
def is_modell(study_name: str, n: int) -> bool:
    patt = rf"_Modell_{n}(?!\d)"
    return re.search(patt, study_name) is not None


# --- Split-Methoden statt Versuchspläne ---
SPLIT_COLOR_MAP = {
    "KS":     "#eef7ff",
    "DUPLEX": "#fff3e0",
    "SPlit":  "#e8f5e9",
    "SPXY":   "#fce4ec",
}
SPLIT_ORDER = ["KS", "DUPLEX", "SPlit", "SPXY"]


def get_split_method(study_name: str) -> str:
    m = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name)
    return m.group(1) if m else "Other"


def get_submodel(study_name: str) -> str | None:
    """Extrahiert bei Modell_1.x das 'x'."""
    m = re.search(r"Modell_1\.(\d)", study_name)
    return m.group(1) if m else None


def shorten_name(study_name: str) -> str:
    """
    Kurzname ohne Versuchsplan: Modell + Split + Seed
    z.B. '..._Modell_1.3_KS_..._seed_0' -> '1.3_KS_0'
    """
    model = re.search(r"Modell_([\d.]+)", study_name)
    seed  = re.search(r"seed_(\d+)", study_name)
    split = get_split_method(study_name)

    mdl = model.group(1) if model else "?"
    sd  = seed.group(1) if seed else "?"
    return f"{mdl}_{split}_{sd}"


def fixed_plan_for_model(n: int) -> str:
    """
    Modell 1 -> Halton
    Modell 2 -> LHS
    """
    return "Halton" if n == 1 else "LHS"

def split_rank(split: str) -> int:
    try:
        return SPLIT_ORDER.index(split)
    except ValueError:
        return 999

def modell_key_from_short(short: str):
    """
    short = '1.3_KS_0' oder '2_SPXY_42'
    -> (major, minor)
    """
    first = short.split("_", 1)[0]  # '1.3' oder '2'
    parts = first.split(".")
    major = int(parts[0]) if parts[0].isdigit() else 999
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)

def apply_consistent_sort(study_names, split_names, seeds, *arrays):
    """
    Sortiert alles konsistent nach:
      split_order -> model/sub -> seed
    Gibt alle Arrays in gleicher Reihenfolge zurück.
    """
    idx = list(range(len(study_names)))
    idx.sort(key=lambda i: (split_rank(split_names[i]),
                            modell_key_from_short(study_names[i]),
                            int(seeds[i]) if seeds[i] is not None else 999999))

    def reorder(x):
        return [x[i] for i in idx]

    out = [reorder(study_names), reorder(split_names), reorder(seeds)]
    for a in arrays:
        out.append(reorder(a))
    return out

def auswertung_modell(n: int):
    study_names, rmse_train, rmse_val, rmse_test = [], [], [], []
    r2_train, r2_val, r2_test = [], [], []
    split_names = []
    seeds = []

    FIXED_PLAN = fixed_plan_for_model(n)

    # --- Daten einsammeln ---
    for study in sorted(os.listdir(base_dir)):
        study_path = os.path.join(base_dir, study)
        if not os.path.isdir(study_path):
            continue
        if not is_modell(study, n):
            continue

        # ✅ Filter: Modell 1 nur Halton, Modell 2 nur LHS
        if f"_{FIXED_PLAN}_" not in study:
            continue

        excel_files = [f for f in os.listdir(study_path) if f.endswith(".xlsx")]
        if not excel_files:
            continue

        excel_path = os.path.join(study_path, sorted(excel_files)[-1])
        df = pd.read_excel(excel_path)

        if "state" not in df.columns or "value" not in df.columns:
            continue

        df_complete = df[df["state"] == "COMPLETE"]
        if df_complete.empty:
            continue

        best_idx = df_complete["value"].idxmin()
        best_trial = df_complete.loc[best_idx]

        if "number" in df_complete.columns:
            trial_num = str(int(best_trial["number"]))
        elif "trial_id" in df_complete.columns:
            trial_num = str(best_trial["trial_id"])
        else:
            trial_num = str(best_trial.iloc[0])

        metrics_dir = os.path.join(study_path, "metrics")
        if not os.path.isdir(metrics_dir):
            continue

        sub = get_submodel(study) if n == 1 else None

        metrics_path = None

        # ✅ Reeval-Pfade: Plan wird nicht mehr geparst, sondern FIXED_PLAN genutzt
        if n == 1 and sub is not None:
            candidate = os.path.join(
                metrics_dir,
                f"metrics_{trial_num}__Reeval_{FIXED_PLAN}_Modell_1.{sub}.csv"
            )
            if os.path.isfile(candidate):
                metrics_path = candidate

        elif n == 2:
            candidate = os.path.join(
                metrics_dir,
                f"metrics_{trial_num}__Reeval_{FIXED_PLAN}_Modell_2.csv"
            )
            if os.path.isfile(candidate):
                metrics_path = candidate

        # Fallback: irgendeine passende csv
        if metrics_path is None:
            csv_candidates = [f for f in os.listdir(metrics_dir)
                              if f.endswith(".csv") and trial_num in f]
            if not csv_candidates:
                csv_candidates = [f for f in os.listdir(metrics_dir) if f.endswith(".csv")]
                if not csv_candidates:
                    continue
            metrics_path = os.path.join(metrics_dir, sorted(csv_candidates)[-1])

        metrics = read_metrics(metrics_path)

        rmse_train.append(metrics['train']['rmse'])
        rmse_val.append(metrics['validation']['rmse'])
        rmse_test.append(metrics['test']['rmse'])
        r2_train.append(metrics['train']['r2'])
        r2_val.append(metrics['validation']['r2'])
        r2_test.append(metrics['test']['r2'])

        m_seed = re.search(r"seed_(\d+)", study)
        seed = int(m_seed.group(1)) if m_seed else None
        study_names.append(shorten_name(study))
        split_names.append(get_split_method(study))
        seeds.append(seed)

        # ✅ Konsistente Sortierung wie in deinen anderen Auswertungen:
        study_names, split_names, seeds, rmse_train, rmse_val, rmse_test, r2_train, r2_val, r2_test = apply_consistent_sort(
            study_names, split_names, seeds,
            rmse_train, rmse_val, rmse_test,
            r2_train, r2_val, r2_test
        )

    if not study_names:
        print(f"Keine Studies für '_Modell_{n}' gefunden (Filter: {FIXED_PLAN}).")
        return

    subfolder = f"Modell_{n}_eigene_Testdaten"

    # Helper für Split-Statistiken
    def aggregate_per_split(values_dict):
        groups, means, stds = [], [], []
        for s in SPLIT_ORDER:
            if s in values_dict:
                vals = np.array(values_dict[s], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                groups.append(s)
                means.append(vals.mean())
                stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        return groups, means, stds

    # =========================
    # 1) RMSE Scatter (Train/Val/Test)
    # =========================
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(split_names):
        color = SPLIT_COLOR_MAP.get(s, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)
    ax.scatter(study_names, rmse_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, rmse_val, marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, rmse_test, marker="o", label="Test", zorder=1, color="#717E86")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.set_title(f"RMSE aller besten Modelle (Modell {n}.*) – {FIXED_PLAN}")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_eigene_Daten_{n}.png", subfolder)

    # =========================
    # 2) R² Scatter (Train/Val/Test)
    # =========================
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(split_names):
        color = SPLIT_COLOR_MAP.get(s, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)
    ax.scatter(study_names, r2_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, r2_val, marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, r2_test, marker="o", label="Test", zorder=1, color="#717E86")
    ax.set_ylabel("R²")
    ax.legend()
    ax.set_title(f"R² aller besten Modelle (Modell {n}.*) – {FIXED_PLAN}")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_eigene_Daten_{n}.png", subfolder)

    # =========================
    # 3) RMSE (Train/Val/Test) pro Split – gruppierte Balken
    # =========================
    split_to_rmse_train = {}
    split_to_rmse_val   = {}
    split_to_rmse_test  = {}

    for s, rm_tr, rm_v, rm_te in zip(split_names, rmse_train, rmse_val, rmse_test):
        if s not in SPLIT_COLOR_MAP:
            continue
        if not pd.isna(rm_tr):
            split_to_rmse_train.setdefault(s, []).append(rm_tr)
        if not pd.isna(rm_v):
            split_to_rmse_val.setdefault(s, []).append(rm_v)
        if not pd.isna(rm_te):
            split_to_rmse_test.setdefault(s, []).append(rm_te)

    groups, mean_tr, std_tr = aggregate_per_split(split_to_rmse_train)
    _,      mean_v,  std_v  = aggregate_per_split(split_to_rmse_val)
    _,      mean_te, std_te = aggregate_per_split(split_to_rmse_test)

    if groups:
        x = np.arange(len(groups))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr, width, yerr=std_tr, capsize=4,
               label="Train", color="#95BB20", edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,  capsize=4,
               label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te, capsize=4,
               label="Test", color="#717E86", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE (Train/Val/Test) pro Aufteilungsmethode (Modell {n}.*) – {FIXED_PLAN}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_TrainValTest_Modell_eigene_Daten_{n}_by_split.png", subfolder)

    # =========================
    # 4) R² (Train/Val/Test) pro Split – gruppierte Balken
    # =========================
    split_to_r2_train = {}
    split_to_r2_val   = {}
    split_to_r2_test  = {}

    for s, r2_tr, r2_v, r2_te in zip(split_names, r2_train, r2_val, r2_test):
        if s not in SPLIT_COLOR_MAP:
            continue
        if not pd.isna(r2_tr):
            split_to_r2_train.setdefault(s, []).append(r2_tr)
        if not pd.isna(r2_v):
            split_to_r2_val.setdefault(s, []).append(r2_v)
        if not pd.isna(r2_te):
            split_to_r2_test.setdefault(s, []).append(r2_te)

    groups, mean_tr, std_tr = aggregate_per_split(split_to_r2_train)
    _,      mean_v,  std_v  = aggregate_per_split(split_to_r2_val)
    _,      mean_te, std_te = aggregate_per_split(split_to_r2_test)

    if groups:
        x = np.arange(len(groups))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr, width, yerr=std_tr, capsize=4,
               label="Train", color="#95BB20", edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,  capsize=4,
               label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te, capsize=4,
               label="Test", color="#717E86", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("R²")
        ax.set_title(f"R² (Train/Val/Test) pro Aufteilungsmethode (Modell {n}.*) – {FIXED_PLAN}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_TrainValTest_Modell_eigene_Daten_{n}_by_split.png", subfolder)

    # =========================
    # 5) RMSE(Test) Scatter sortiert
    # =========================
    order_rmse = np.argsort(rmse_test)
    sorted_names = [study_names[i] for i in order_rmse]
    sorted_rmse  = [rmse_test[i] for i in order_rmse]
    sorted_splits = [split_names[i] for i in order_rmse]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(sorted_splits):
        color = SPLIT_COLOR_MAP.get(s, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)
    ax.scatter(sorted_names, sorted_rmse, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("RMSE Test")
    ax.set_title(f"RMSE (Test) über alle besten Modelle (Modell {n}.*) – {FIXED_PLAN}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_Test_Modell_eigene_Daten_{n}.png", subfolder)

    # =========================
    # 6) R²(Test) Scatter sortiert
    # =========================
    # NaNs ans Ende sortieren
    r2_for_sort = [(-x) if pd.notna(x) else np.inf for x in r2_test]
    order_r2 = np.argsort(r2_for_sort)

    sorted_names = [study_names[i] for i in order_r2]
    sorted_r2    = [r2_test[i] for i in order_r2]
    sorted_splits = [split_names[i] for i in order_r2]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(sorted_splits):
        color = SPLIT_COLOR_MAP.get(s, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)
    ax.scatter(sorted_names, sorted_r2, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("R² Test")
    ax.set_title(f"R² (Test) über alle besten Modelle (Modell {n}.*) – {FIXED_PLAN}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_Test_Modell_eigene_Daten_{n}.png", subfolder)

    # =========================
    # 7) RMSE(Test) pro Split – Balken
    # =========================
    split_to_rmse = {}
    for rmse, s in zip(rmse_test, split_names):
        if pd.isna(rmse):
            continue
        if s not in SPLIT_COLOR_MAP:
            continue
        split_to_rmse.setdefault(s, []).append(rmse)

    groups, means, stds = aggregate_per_split(split_to_rmse)
    if groups:
        x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [SPLIT_COLOR_MAP.get(g, "#cccccc") for g in groups]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("RMSE Test")
        ax.set_title(f"RMSE (Test) pro Aufteilungsmethode (Modell {n}.*) – {FIXED_PLAN}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_Test_Modell_eigene_Daten_{n}_by_split.png", subfolder)

    # =========================
    # 8) R²(Test) pro Split – Balken
    # =========================
    split_to_r2 = {}
    for r2v, s in zip(r2_test, split_names):
        if pd.isna(r2v):
            continue
        if s not in SPLIT_COLOR_MAP:
            continue
        split_to_r2.setdefault(s, []).append(r2v)

    groups, means, stds = aggregate_per_split(split_to_r2)
    if groups:
        x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [SPLIT_COLOR_MAP.get(g, "#cccccc") for g in groups]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("R² Test")
        ax.set_title(f"R² (Test) pro Aufteilungsmethode (Modell {n}.*) – {FIXED_PLAN}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_Test_Modell_eigene_Daten_{n}_by_split.png", subfolder)

    print(f"Plots für Modell {n} gespeichert unter: {os.path.join(output_dir, subfolder)}")


auswertung_modell(1)
auswertung_modell(2)