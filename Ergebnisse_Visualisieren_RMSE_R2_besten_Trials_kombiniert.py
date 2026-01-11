import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "Ergebnisse_Teil_1"
output_dir = "Ergebnisplots_Visualization_Metriken_kombiniert"
os.makedirs(output_dir, exist_ok=True)


def save_plot(fig, filename, subfolder):
    """Hilfsfunktion zum Speichern jedes Plots."""
    save_path = os.path.join(output_dir, subfolder)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_metrics(metrics_path):
    """
    Liest eine metrics_*.csv ein und gibt ein Dict der Form zurück:
    {
      'train':      {'rmse': ..., 'r2': ...},
      'validation': {'rmse': ..., 'r2': ...},
      'test':       {'rmse': ..., 'r2': ...}
    }
    """
    m = pd.read_csv(metrics_path)
    m.columns = [str(c).strip().lower() for c in m.columns]
    out = {
        'train':      {'rmse': np.nan, 'r2': np.nan},
        'validation': {'rmse': np.nan, 'r2': np.nan},
        'test':       {'rmse': np.nan, 'r2': np.nan},
    }

    # Fall 1: dataset / rmse / r2 Spalten (deine aktuelle Struktur)
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

    # Fall 2: rmse_train, rmse_validation, rmse_test etc.
    def get(col):
        return m[col].iloc[-1] if col in m.columns and not m[col].empty else np.nan

    for ds in out:
        out[ds]['rmse'] = get(f'rmse_{ds}')
        out[ds]['r2']   = get(f'r2_{ds}')
    return out


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
    Extrahiert aus:
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


COLOR_MAP = {
    "Halton":  "#eef7ff",
    "Sobol":   "#fff3e0",
    "Taguchi": "#e8f5e9",
    "LHS":     "#fce4ec",
}


def get_plan(study_name: str) -> str:
    m = re.search(r"_(Halton|Sobol|Taguchi|LHS)_", study_name)
    return m.group(1) if m else "Other"


def get_submodel(study_name: str):
    """
    Extrahiert bei Modell_1.x das 'x', z.B.
    'Study_..._Halton_Modell_1.3_KS_Holdout_seed_0' -> '3'
    """
    m = re.search(r"Modell_1\.(\d)", study_name)
    return m.group(1) if m else None


def auswertung_modell(n: int):
    study_names = []
    method_names = []

    rmse_train = []
    rmse_val = []
    rmse_test_global = []
    rmse_test_own = []

    r2_train = []
    r2_val = []
    r2_test_global = []
    r2_test_own = []

    # --- Daten einsammeln ---
    for study in sorted(os.listdir(base_dir)):
        study_path = os.path.join(base_dir, study)
        if not os.path.isdir(study_path):
            continue
        if not is_modell(study, n):
            continue

        # Study-Excel finden (Optuna-Study)
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

        # ---- 1) Globale Metriken: metrics_{trial}.csv ----
        metrics_path_global = os.path.join(metrics_dir, f"metrics_{trial_num}.csv")
        if not os.path.isfile(metrics_path_global):
            # Fallback: irgendeine passende metrics-CSV für diesen Trial
            csv_candidates = [f for f in os.listdir(metrics_dir)
                              if f.endswith(".csv") and trial_num in f]
            if not csv_candidates:
                csv_candidates = [f for f in os.listdir(metrics_dir) if f.endswith(".csv")]
                if not csv_candidates:
                    continue
            metrics_path_global = os.path.join(metrics_dir, sorted(csv_candidates)[-1])

        metrics_global = read_metrics(metrics_path_global)

        # ---- 2) Reeval-Metriken: eigene Testdaten ----
        plan = get_plan(study)
        sub  = get_submodel(study) if n == 1 else None

        metrics_path_own = None
        if n == 1 and sub is not None:
            cand = os.path.join(
                metrics_dir,
                f"metrics_{trial_num}__Reeval_{plan}_Modell_1.{sub}.csv"
            )
            if os.path.isfile(cand):
                metrics_path_own = cand
        elif n == 2:
            cand = os.path.join(
                metrics_dir,
                f"metrics_{trial_num}__Reeval_{plan}_Modell_2.csv"
            )
            if os.path.isfile(cand):
                metrics_path_own = cand

        metrics_own = None
        if metrics_path_own is not None and os.path.isfile(metrics_path_own):
            metrics_own = read_metrics(metrics_path_own)

        # ---- Werte einsammeln ----
        rmse_train.append(metrics_global['train']['rmse'])
        rmse_val.append(metrics_global['validation']['rmse'])
        rmse_test_global.append(metrics_global['test']['rmse'])

        r2_train.append(metrics_global['train']['r2'])
        r2_val.append(metrics_global['validation']['r2'])
        r2_test_global.append(metrics_global['test']['r2'])

        if metrics_own is not None:
            rmse_test_own.append(metrics_own['test']['rmse'])
            r2_test_own.append(metrics_own['test']['r2'])
        else:
            rmse_test_own.append(np.nan)
            r2_test_own.append(np.nan)

        study_names.append(shorten_name(study))
        method_names.append(plan)

    if not study_names:
        print(f"Keine Studies für '_Modell_{n}' gefunden.")
        return

    subfolder = f"Modell_{n}"

    def aggregate_per_plan(values_dict):
        plans, means, stds = [], [], []
        for plan in ["Halton", "LHS", "Sobol", "Taguchi"]:
            if plan not in values_dict:
                continue
            vals = np.array(values_dict[plan], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            plans.append(plan)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        return plans, means, stds

    # ================================
    # Plot 1) RMSE Train / Val / Test(global) + Test(eigene Daten)
    # ================================
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(method_names):
        color = COLOR_MAP.get(method, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)

    ax.scatter(study_names, rmse_train, marker="o",
               label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, rmse_val, marker="o",
               label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, rmse_test_global, marker="o",
               label="Test global", zorder=1, color="#717E86")
    ax.scatter(study_names, rmse_test_own, marker="o",
               label="Test eigene Daten", zorder=1, color="#0098AD")

    ax.set_ylim(top=20)
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.set_title(f"RMSE aller besten Modelle (Modell {n}.*)\nGlobales Testset vs. eigene Testdaten")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_{n}_kombiniert.png", subfolder)

    # ================================
    # Plot 2) R² Train / Val / Test(global) + Test(eigene Daten)
    # ================================
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(method_names):
        color = COLOR_MAP.get(method, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)

    ax.scatter(study_names, r2_train, marker="o",
               label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, r2_val, marker="o",
               label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, r2_test_global, marker="o",
               label="Test global", zorder=1, color="#717E86")
    ax.scatter(study_names, r2_test_own, marker="o",
               label="Test eigene Daten", zorder=1, color="#0098AD")

    ax.set_ylabel("R²")
    ax.legend()
    ax.set_title(f"R² aller besten Modelle (Modell {n}.*)\nGlobales Testset vs. eigene Testdaten")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_{n}_kombiniert.png", subfolder)

    # ================================
    # Plot 3) RMSE (Train/Val/Test global/Test eigene) pro Versuchsplan – gruppierte Balken
    # ================================
    plan_to_rmse_train = {}
    plan_to_rmse_val = {}
    plan_to_rmse_glob = {}
    plan_to_rmse_own = {}

    for plan, rm_tr, rm_v, rm_g, rm_o in zip(method_names, rmse_train, rmse_val, rmse_test_global, rmse_test_own):
        if plan not in COLOR_MAP:
            continue
        if not pd.isna(rm_tr):
            plan_to_rmse_train.setdefault(plan, []).append(rm_tr)
        if not pd.isna(rm_v):
            plan_to_rmse_val.setdefault(plan, []).append(rm_v)
        if not pd.isna(rm_g):
            plan_to_rmse_glob.setdefault(plan, []).append(rm_g)
        if not pd.isna(rm_o):
            plan_to_rmse_own.setdefault(plan, []).append(rm_o)

    plans, mean_tr, std_tr = aggregate_per_plan(plan_to_rmse_train)
    _, mean_v, std_v = aggregate_per_plan(plan_to_rmse_val)
    _, mean_g, std_g = aggregate_per_plan(plan_to_rmse_glob)
    _, mean_o, std_o = aggregate_per_plan(plan_to_rmse_own)

    if plans:
        x = np.arange(len(plans))
        width = 0.20
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.bar(x - 1.5 * width, mean_tr, width, yerr=std_tr, capsize=4,
               label="Train", color="#95BB20", edgecolor="black")
        ax.bar(x - 0.5 * width, mean_v, width, yerr=std_v, capsize=4,
               label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + 0.5 * width, mean_g, width, yerr=std_g, capsize=4,
               label="Test global", color="#717E86", edgecolor="black")
        ax.bar(x + 1.5 * width, mean_o, width, yerr=std_o, capsize=4,
               label="Test eigene Daten", color="#0098AD", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(plans)
        ax.set_ylabel("RMSE")
        ax.legend(loc="upper right")
        ax.set_title(f"RMSE (Train/Val/Test global/Test eigene) pro Versuchsplan (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_TrainValTest_all_Modell_{n}_by_plan_kombiniert.png", subfolder)

    # ================================
    # Plot 4) R² (Train/Val/Test global/Test eigene) pro Versuchsplan – gruppierte Balken
    # ================================
    plan_to_r2_train = {}
    plan_to_r2_val = {}
    plan_to_r2_glob = {}
    plan_to_r2_own = {}

    for plan, r2_tr, r2_v, r2_g, r2_o in zip(method_names, r2_train, r2_val, r2_test_global, r2_test_own):
        if plan not in COLOR_MAP:
            continue
        if not pd.isna(r2_tr):
            plan_to_r2_train.setdefault(plan, []).append(r2_tr)
        if not pd.isna(r2_v):
            plan_to_r2_val.setdefault(plan, []).append(r2_v)
        if not pd.isna(r2_g):
            plan_to_r2_glob.setdefault(plan, []).append(r2_g)
        if not pd.isna(r2_o):
            plan_to_r2_own.setdefault(plan, []).append(r2_o)

    plans, mean_tr, std_tr = aggregate_per_plan(plan_to_r2_train)
    _, mean_v, std_v = aggregate_per_plan(plan_to_r2_val)
    _, mean_g, std_g = aggregate_per_plan(plan_to_r2_glob)
    _, mean_o, std_o = aggregate_per_plan(plan_to_r2_own)

    if plans:
        x = np.arange(len(plans))
        width = 0.20
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.bar(x - 1.5 * width, mean_tr, width, yerr=std_tr, capsize=4,
               label="Train", color="#95BB20", edgecolor="black")
        ax.bar(x - 0.5 * width, mean_v, width, yerr=std_v, capsize=4,
               label="Validation", color="#00354E", edgecolor="black")
        ax.bar(x + 0.5 * width, mean_g, width, yerr=std_g, capsize=4,
               label="Test global", color="#717E86", edgecolor="black")
        ax.bar(x + 1.5 * width, mean_o, width, yerr=std_o, capsize=4,
               label="Test eigene Daten", color="#0098AD", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(plans)
        ax.set_ylabel("R²")
        ax.legend()
        ax.set_title(f"R² (Train/Val/Test global/Test eigene) pro Versuchsplan (Modell {n}.*)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_TrainValTest_all_Modell_{n}_by_plan_kombiniert.png", subfolder)

    print(f"Plots für Modell {n} gespeichert unter: {os.path.join(output_dir, subfolder)}")


if __name__ == "__main__":
    #auswertung_modell(1)
    auswertung_modell(2)