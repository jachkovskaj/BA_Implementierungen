import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "Ergebnisse_Teil_1"
output_dir = "Ergebnisplots_Visualization_Metriken"
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
    out = {'train': {'rmse': np.nan, 'r2': np.nan},
           'validation': {'rmse': np.nan, 'r2': np.nan},
           'test': {'rmse': np.nan, 'r2': np.nan}}

    if {'dataset','rmse','r2'}.issubset(m.columns):
        norm = {'train':'train','training':'train',
                'valid':'validation','validation':'validation','val':'validation',
                'test':'test','testing':'test'}
        m['dataset'] = m['dataset'].astype(str).str.strip().str.lower().map(norm)
        g = (m.dropna(subset=['dataset'])
               .groupby('dataset', sort=False)[['rmse','r2']]
               .agg('last'))
        for ds in out:
            if ds in g.index:
                out[ds]['rmse'] = float(g.loc[ds,'rmse'])
                out[ds]['r2']   = float(g.loc[ds,'r2'])
        return out

    def get(col):
        return m[col].iloc[-1] if col in m.columns and not m[col].empty else np.nan
    for ds in out:
        out[ds]['rmse'] = get(f'rmse_{ds}')
        out[ds]['r2']   = get(f'r2_{ds}')
    return out

# Helfer: Ordnername gehört zu Modell N?
def is_modell(study_name: str, n: int) -> bool:
    """
    True, wenn der Ordnername ein '_Modell_<n>' enthält, gefolgt von
    Nicht-Ziffer (z. B. '.', '_', Ende) — deckt '_Modell_1.', '_Modell_1.1',
    '_Modell_2', '_Modell_2_seed' etc. ab, vermeidet aber '_Modell_11'.
    """
    patt = rf"_Modell_{n}(?!\d)"   # nächste Stelle ist KEINE Ziffer
    return re.search(patt, study_name) is not None

def shorten_name(study_name: str) -> str:
    """
    Extrahiert aus einem vollständigen Studynamen wie:
    'Study_15_10_2025_Halton_Modell_1.1_KS_Holdout_seed_0'
    -> 'Halton_1.1_0'
    """
    method = re.search(r"_(Halton|Sobol|Taguchi|LHS)_", study_name)
    model  = re.search(r"Modell_([\d.]+)", study_name)
    seed   = re.search(r"seed_(\d+)", study_name)

    mth = method.group(1) if method else "?"
    mdl = model.group(1) if model else "?"
    sd  = seed.group(1) if seed else "?"
    return f"{mth}_{mdl}_{sd}"

# Mapping: Versuchsplan -> Hintergrundfarbe
COLOR_MAP = {
    "Halton":  "#eef7ff",
    "Sobol":   "#fff3e0",
    "Taguchi": "#e8f5e9",
    "LHS":     "#fce4ec",
}

def get_plan(study_name: str) -> str:
    m = re.search(r"_(Halton|Sobol|Taguchi|LHS)_", study_name)
    return m.group(1) if m else "Other"

# Hilfsfunktion für Visualisierung mit eigenen Daten
def get_submodel(study_name: str) -> str | None:
    """
    Extrahiert bei Modell_1.x das 'x', z.B.
    'Study_..._Halton_Modell_1.3_KS_Holdout_seed_0' -> '3'
    """
    m = re.search(r"Modell_1\.(\d)", study_name)
    return m.group(1) if m else None

# Auswertung für Modell-N
def auswertung_modell(n: int):
    study_names, rmse_train, rmse_val, rmse_test = [], [], [], []
    r2_train, r2_val, r2_test = [], [], []
    method_names = []

    # --- Daten einsammeln ---
    for study in sorted(os.listdir(base_dir)):
        study_path = os.path.join(base_dir, study)
        if not os.path.isdir(study_path):
            continue
        if not is_modell(study, n):
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

        # Ordner mit Metriken
        metrics_dir = os.path.join(study_path, "metrics")
        if not os.path.isdir(metrics_dir):
            # falls es keinen metrics-Ordner gibt, nächsten Study nehmen
            continue

        # **GLOBALER** metrics-File: metrics_{trial}.csv
        metrics_path = os.path.join(metrics_dir, f"metrics_{trial_num}.csv")
        if not os.path.isfile(metrics_path):
            # Fallback wie bisher: irgendeine passende metrics-CSV für diesen Trial
            csv_candidates = [f for f in os.listdir(metrics_dir)
                              if f.endswith(".csv") and trial_num in f]
            if not csv_candidates:
                csv_candidates = [f for f in os.listdir(metrics_dir) if f.endswith(".csv")]
                if not csv_candidates:
                    continue
            metrics_path = os.path.join(metrics_dir, sorted(csv_candidates)[-1])

        # Jetzt wirklich einlesen
        metrics = read_metrics(metrics_path)

        rmse_train.append(metrics['train']['rmse'])
        rmse_val.append(metrics['validation']['rmse'])
        rmse_test.append(metrics['test']['rmse'])
        r2_train.append(metrics['train']['r2'])
        r2_val.append(metrics['validation']['r2'])
        r2_test.append(metrics['test']['r2'])

        study_names.append(shorten_name(study))
        method_names.append(get_plan(study))

    if not study_names:
        print(f"Keine Studies für '_Modell_{n}' gefunden.")
        return

    # Subfolder
    subfolder = f"Modell_{n}"

    # Helper für Plan-Statistiken
    def aggregate_per_plan(values_dict):
        plans, means, stds = [], [], []
        for plan in ["Halton", "LHS", "Sobol", "Taguchi"]:
            if plan in values_dict:
                vals = np.array(values_dict[plan], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                plans.append(plan)
                means.append(vals.mean())
                stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        return plans, means, stds

    # 1) RMSE Train/Valid/Test
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(method_names):
        color = COLOR_MAP.get(method, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)
    ax.scatter(study_names, rmse_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, rmse_val, marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, rmse_test, marker="o", label="Test", zorder=1, color="#717E86")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.set_title(f"RMSE aller besten Modelle (Modell {n}.*)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_{n}.png", subfolder)

    # 2) R² Train/Valid/Test
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(method_names):
        color = COLOR_MAP.get(method, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)
    ax.scatter(study_names, r2_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, r2_val, marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, r2_test, marker="o", label="Test", zorder=1, color="#717E86")
    ax.set_ylabel("R²")
    ax.legend()
    ax.set_title(f"R² aller besten Modelle (Modell {n}.*)")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_{n}.png", subfolder)

    # ================================
    # 3) RMSE (Train/Val/Test) pro Versuchsplan – gruppierte Balken
    # ================================
    plan_to_rmse_train = {}
    plan_to_rmse_val   = {}
    plan_to_rmse_test  = {}

    for plan, rm_tr, rm_v, rm_te in zip(method_names, rmse_train, rmse_val, rmse_test):
        if plan not in COLOR_MAP:
            continue
        if not pd.isna(rm_tr):
            plan_to_rmse_train.setdefault(plan, []).append(rm_tr)
        if not pd.isna(rm_v):
            plan_to_rmse_val.setdefault(plan, []).append(rm_v)
        if not pd.isna(rm_te):
            plan_to_rmse_test.setdefault(plan, []).append(rm_te)

    plans, mean_tr, std_tr = aggregate_per_plan(plan_to_rmse_train)
    _,     mean_v,  std_v  = aggregate_per_plan(plan_to_rmse_val)
    _,     mean_te, std_te = aggregate_per_plan(plan_to_rmse_test)

    if plans:
        x = np.arange(len(plans))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr,  width, yerr=std_tr,  capsize=4,
               label="Train",      color="#95BB20", edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,   capsize=4,
               label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te,  capsize=4,
               label="Test",       color="#717E86", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(plans)
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE (Train/Val/Test) pro Versuchsplan (Modell {n}.*)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_TrainValTest_Modell_{n}_by_plan.png", subfolder)

    # ================================
    # 4) R² (Train/Val/Test) pro Versuchsplan – gruppierte Balken
    # ================================
    plan_to_r2_train = {}
    plan_to_r2_val   = {}
    plan_to_r2_test  = {}

    for plan, r2_tr, r2_v, r2_te in zip(method_names, r2_train, r2_val, r2_test):
        if plan not in COLOR_MAP:
            continue
        if not pd.isna(r2_tr):
            plan_to_r2_train.setdefault(plan, []).append(r2_tr)
        if not pd.isna(r2_v):
            plan_to_r2_val.setdefault(plan, []).append(r2_v)
        if not pd.isna(r2_te):
            plan_to_r2_test.setdefault(plan, []).append(r2_te)

    plans, mean_tr, std_tr = aggregate_per_plan(plan_to_r2_train)
    _,     mean_v,  std_v  = aggregate_per_plan(plan_to_r2_val)
    _,     mean_te, std_te = aggregate_per_plan(plan_to_r2_test)

    if plans:
        x = np.arange(len(plans))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr,  width, yerr=std_tr,  capsize=4,
               label="Train",      color="#95BB20", edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,   capsize=4,
               label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te,  capsize=4,
               label="Test",       color="#717E86", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(plans)
        ax.set_ylabel("R²")
        ax.set_title(f"R² (Train/Val/Test) pro Versuchsplan (Modell {n}.*)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_TrainValTest_Modell_{n}_by_plan.png", subfolder)

    # ================================
    # 5) RMSE(Test) Scatter über alle Studies (nach RMSE sortiert)
    # ================================
    order_rmse = np.argsort(rmse_test)
    sorted_names   = [study_names[i] for i in order_rmse]
    sorted_rmse    = [rmse_test[i]   for i in order_rmse]
    sorted_methods = [method_names[i] for i in order_rmse]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(sorted_methods):
        color = COLOR_MAP.get(method, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)

    ax.scatter(sorted_names, sorted_rmse, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("RMSE Test")
    ax.set_title(f"RMSE (Test) über alle besten Modelle (Modell {n}.*)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_Test_Modell_{n}.png", subfolder)

    # ================================
    # 6) R²(Test) Scatter über alle Studies (nach R² sortiert)
    # ================================
    order_r2 = np.argsort([-x if pd.notna(x) else np.nan for x in r2_test])
    sorted_names   = [study_names[i] for i in order_r2]
    sorted_r2      = [r2_test[i]     for i in order_r2]
    sorted_methods = [method_names[i] for i in order_r2]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(sorted_methods):
        color = COLOR_MAP.get(method, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)

    ax.scatter(sorted_names, sorted_r2, marker="o", zorder=1, color="#717E86")
    ax.set_ylabel("R² Test")
    ax.set_title(f"R² (Test) über alle besten Modelle (Modell {n}.*)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_Test_Modell_{n}.png", subfolder)

    # ================================
    # 7) RMSE(Test) pro Versuchsplan – Balken
    # ================================
    plan_to_rmse = {}
    for rmse, method in zip(rmse_test, method_names):
        if pd.isna(rmse):
            continue
        if method not in COLOR_MAP:
            continue
        plan_to_rmse.setdefault(method, []).append(rmse)

    plans, means, stds = aggregate_per_plan(plan_to_rmse)
    if plans:
        x = np.arange(len(plans))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [COLOR_MAP.get(p, "#cccccc") for p in plans]

        ax.bar(x, means, yerr=stds, capsize=5,
               color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(plans)
        ax.set_ylabel("RMSE Test")
        ax.set_title(f"RMSE (Test) pro Versuchsplan (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_Test_Modell_{n}_by_plan.png", subfolder)

    # ================================
    # 8) R²(Test) pro Versuchsplan – Balken
    # ================================
    plan_to_r2 = {}
    for r2_val, method in zip(r2_test, method_names):
        if pd.isna(r2_val):
            continue
        if method not in COLOR_MAP:
            continue
        plan_to_r2.setdefault(method, []).append(r2_val)

    plans, means, stds = aggregate_per_plan(plan_to_r2)
    if plans:
        x = np.arange(len(plans))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [COLOR_MAP.get(p, "#cccccc") for p in plans]

        ax.bar(x, means, yerr=stds, capsize=5,
               color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(plans)
        ax.set_ylabel("R² Test")
        ax.set_title(f"R² (Test) pro Versuchsplan (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_Test_Modell_{n}_by_plan.png", subfolder)

    print(f"Plots für Modell {n} gespeichert unter: {os.path.join(output_dir, subfolder)}")

auswertung_modell(1)
auswertung_modell(2)