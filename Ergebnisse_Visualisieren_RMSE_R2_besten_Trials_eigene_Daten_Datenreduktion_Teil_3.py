import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Konfiguration
# ============================================================
base_dir = "Ergebnisse_Teil_3"
output_dir = "Ergebnisplots_Visualization_Metriken_Datenreduktion_eigene_Daten"
os.makedirs(output_dir, exist_ok=True)

REDUCTION_ORDER = ["CBIR", "RegMix"]
REDUCTION_COLOR_MAP = {
    "CBIR":  "#eef7ff",  # blau
    "RegMix":"#fff3e0",  # gelb
}

# Fixierungen wie bei dir
PLAN_FIX  = {1: "Halton", 2: "LHS"}
SPLIT_FIX = {1: "SPXY",   2: "KS"}


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


def get_submodel(study_name: str) -> str | None:
    """Extrahiert bei Modell_1.x das 'x'."""
    m = re.search(r"Modell_1\.(\d)", study_name)
    return m.group(1) if m else None


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
    m = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name, re.IGNORECASE)
    if not m:
        return "Other"
    s = m.group(1)
    # Normierung auf deine Schreibweise
    mp = {"ks": "KS", "duplex": "DUPLEX", "split": "SPlit", "spxy": "SPXY"}
    return mp.get(s.lower(), s)


def get_reduction_method(study_name: str) -> str:
    m = re.search(r"_(CBIR|RegMix)_", study_name, re.IGNORECASE)
    if not m:
        return "Other"
    r = m.group(1)
    return "CBIR" if r.lower() == "cbir" else "RegMix"


def get_seed_int(study_name: str) -> int | None:
    m = re.search(r"seed_(\d+)", study_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def get_conf_int(study_name: str) -> int | None:
    m = re.search(r"_CONF_(\d+)", study_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def shorten_name(study_name: str) -> str:
    """
    Kurzname:
      Modell + Reduktion + Seed [+ _CONFZAHL]
    z.B. ..._Modell_1.3_..._CBIR_..._seed_0_CONF_85 -> 1.3_CBIR_0_85
    """
    model = re.search(r"Modell_([\d.]+)", study_name)
    mdl = model.group(1) if model else "?"

    red = get_reduction_method(study_name)
    seed = get_seed_int(study_name)
    conf = get_conf_int(study_name)

    sd = str(seed) if seed is not None else "?"
    if conf is not None:
        return f"{mdl}_{red}_{sd}_{conf}"
    return f"{mdl}_{red}_{sd}"


def modell_key_from_short(short: str):
    """
    short beginnt mit '1.3' oder '2'
    -> (major, minor)
    """
    first = short.split("_", 1)[0]
    parts = first.split(".")
    major = int(parts[0]) if parts[0].isdigit() else 999
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)


def apply_consistent_sort(study_names, red_names, seeds, confs, *arrays):
    """
    Sortierung:
      reduction_order -> model/sub -> seed -> conf
    """
    red_rank = {r: i for i, r in enumerate(REDUCTION_ORDER)}

    idx = list(range(len(study_names)))
    idx.sort(key=lambda i: (
        red_rank.get(red_names[i], 999),
        modell_key_from_short(study_names[i]),
        int(seeds[i]) if seeds[i] is not None else 999999,
        int(confs[i]) if confs[i] is not None else 999999,
    ))

    def reorder(x):
        return [x[i] for i in idx]

    out = [reorder(study_names), reorder(red_names), reorder(seeds), reorder(confs)]
    for a in arrays:
        out.append(reorder(a))
    return out


def find_reeval_metrics_path(metrics_dir: str, trial_num: str, fixed_plan: str, n: int, sub: str | None):
    """
     Nutzt Reeval (eigene Testdaten), wie dein Code bisher:
      Modell 1: metrics_{trial}__Reeval_{PLAN}_Modell_1.{sub}.csv
      Modell 2: metrics_{trial}__Reeval_{PLAN}_Modell_2.csv
    Fallback: irgendeine csv mit trial_num (wenn nötig)
    """
    if not os.path.isdir(metrics_dir):
        return None

    metrics_path = None
    if n == 1 and sub is not None:
        candidate = os.path.join(metrics_dir, f"metrics_{trial_num}__Reeval_{fixed_plan}_Modell_1.{sub}.csv")
        if os.path.isfile(candidate):
            metrics_path = candidate
    elif n == 2:
        candidate = os.path.join(metrics_dir, f"metrics_{trial_num}__Reeval_{fixed_plan}_Modell_2.csv")
        if os.path.isfile(candidate):
            metrics_path = candidate

    if metrics_path is not None:
        return metrics_path

    # Fallback: irgendeine passende csv (wie vorher)
    csv_candidates = [f for f in os.listdir(metrics_dir) if f.endswith(".csv") and trial_num in f]
    if not csv_candidates:
        csv_candidates = [f for f in os.listdir(metrics_dir) if f.endswith(".csv")]
        if not csv_candidates:
            return None
    return os.path.join(metrics_dir, sorted(csv_candidates)[-1])


def auswertung_modell(n: int):
    study_names, rmse_train, rmse_val, rmse_test = [], [], [], []
    r2_train, r2_val, r2_test = [], [], []
    red_names = []
    seeds = []
    confs = []

    fixed_plan  = PLAN_FIX.get(n)
    fixed_split = SPLIT_FIX.get(n)

    # --- Daten einsammeln ---
    for study in sorted(os.listdir(base_dir)):
        study_path = os.path.join(base_dir, study)
        if not os.path.isdir(study_path):
            continue
        if not is_modell(study, n):
            continue

        #  Reduktion bestimmen
        red = get_reduction_method(study)
        if red not in REDUCTION_ORDER:
            continue

        #  CBIR nur CONF-Studies
        if red == "CBIR" and "_CONF_" not in study:
            continue

        #  Plan/Split Fix
        if fixed_plan and get_plan(study) != fixed_plan:
            continue
        if fixed_split and get_split_method(study) != fixed_split:
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

        metrics_path = find_reeval_metrics_path(metrics_dir, trial_num, fixed_plan, n, sub)
        if metrics_path is None:
            continue

        metrics = read_metrics(metrics_path)

        rmse_train.append(metrics['train']['rmse'])
        rmse_val.append(metrics['validation']['rmse'])
        rmse_test.append(metrics['test']['rmse'])
        r2_train.append(metrics['train']['r2'])
        r2_val.append(metrics['validation']['r2'])
        r2_test.append(metrics['test']['r2'])

        seed = get_seed_int(study)
        conf = get_conf_int(study)

        study_names.append(shorten_name(study))
        red_names.append(red)
        seeds.append(seed)
        confs.append(conf)

        #  Konsistente Sortierung
        study_names, red_names, seeds, confs, rmse_train, rmse_val, rmse_test, r2_train, r2_val, r2_test = apply_consistent_sort(
            study_names, red_names, seeds, confs,
            rmse_train, rmse_val, rmse_test,
            r2_train, r2_val, r2_test
        )

    if not study_names:
        print(f"Keine Studies für Modell {n} gefunden (Plan={fixed_plan}, Split={fixed_split}).")
        return

    subfolder = f"Modell_{n}_eigene_Testdaten_Datenreduktion"

    def aggregate_per_reduction(values_dict):
        groups, means, stds = [], [], []
        for r in REDUCTION_ORDER:
            if r in values_dict:
                vals = np.array(values_dict[r], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                groups.append(r)
                means.append(vals.mean())
                stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        return groups, means, stds

    # =========================
    # 1) RMSE Scatter (Train/Val/Test)
    # =========================
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(red_names):
        color = REDUCTION_COLOR_MAP.get(r, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)

    ax.scatter(study_names, rmse_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, rmse_val,   marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, rmse_test,  marker="o", label="Test", zorder=1, color="#717E86")

    ax.set_ylabel("RMSE")
    ax.legend()
    ax.set_title(f"RMSE aller besten Modelle (Modell {n}.*) – Datenreduktion (CBIR vs RegMix) – {fixed_plan}/{fixed_split}")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"RMSE_all_Modell_{n}_Datenreduktion_eigene_Daten.png", subfolder)

    # =========================
    # 2) R² Scatter (Train/Val/Test)
    # =========================
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(red_names):
        color = REDUCTION_COLOR_MAP.get(r, "#ffffff")
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7, zorder=0)

    ax.scatter(study_names, r2_train, marker="o", label="Train", zorder=1, color="#95BB20")
    ax.scatter(study_names, r2_val,   marker="o", label="Validation", zorder=1, color="#00354E")
    ax.scatter(study_names, r2_test,  marker="o", label="Test", zorder=1, color="#717E86")

    ax.set_ylabel("R²")
    ax.legend()
    ax.set_title(f"R² aller besten Modelle (Modell {n}.*) – Datenreduktion (CBIR vs RegMix) – {fixed_plan}/{fixed_split}")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    save_plot(fig, f"R2_all_Modell_{n}_Datenreduktion_eigene_Daten.png", subfolder)

    # =========================
    # 3) RMSE (Train/Val/Test) pro Reduktion – gruppierte Balken
    # =========================
    red_to_rmse_train, red_to_rmse_val, red_to_rmse_test = {}, {}, {}
    for r, rm_tr, rm_v, rm_te in zip(red_names, rmse_train, rmse_val, rmse_test):
        if not pd.isna(rm_tr): red_to_rmse_train.setdefault(r, []).append(rm_tr)
        if not pd.isna(rm_v):  red_to_rmse_val.setdefault(r, []).append(rm_v)
        if not pd.isna(rm_te): red_to_rmse_test.setdefault(r, []).append(rm_te)

    groups, mean_tr, std_tr = aggregate_per_reduction(red_to_rmse_train)
    _,      mean_v,  std_v  = aggregate_per_reduction(red_to_rmse_val)
    _,      mean_te, std_te = aggregate_per_reduction(red_to_rmse_test)

    if groups:
        x = np.arange(len(groups))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr, width, yerr=std_tr, capsize=4, label="Train",      edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,  capsize=4, label="Validation", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te, capsize=4, label="Test",       edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE (Train/Val/Test) pro Reduktionsmethode (Modell {n}.*) – {fixed_plan}/{fixed_split}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"RMSE_TrainValTest_Modell_{n}_by_reduction_eigene_Daten.png", subfolder)

    # =========================
    # 4) R² (Train/Val/Test) pro Reduktion – gruppierte Balken
    # =========================
    red_to_r2_train, red_to_r2_val, red_to_r2_test = {}, {}, {}
    for r, r2_tr, r2_v, r2_te in zip(red_names, r2_train, r2_val, r2_test):
        if not pd.isna(r2_tr): red_to_r2_train.setdefault(r, []).append(r2_tr)
        if not pd.isna(r2_v):  red_to_r2_val.setdefault(r, []).append(r2_v)
        if not pd.isna(r2_te): red_to_r2_test.setdefault(r, []).append(r2_te)

    groups, mean_tr, std_tr = aggregate_per_reduction(red_to_r2_train)
    _,      mean_v,  std_v  = aggregate_per_reduction(red_to_r2_val)
    _,      mean_te, std_te = aggregate_per_reduction(red_to_r2_test)

    if groups:
        x = np.arange(len(groups))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width, mean_tr, width, yerr=std_tr, capsize=4, label="Train",      edgecolor="black")
        ax.bar(x,         mean_v,  width, yerr=std_v,  capsize=4, label="Validation", edgecolor="black")
        ax.bar(x + width, mean_te, width, yerr=std_te, capsize=4, label="Test",       edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("R²")
        ax.set_title(f"R² (Train/Val/Test) pro Reduktionsmethode (Modell {n}.*) – {fixed_plan}/{fixed_split}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_plot(fig, f"R2_TrainValTest_Modell_{n}_by_reduction_eigene_Daten.png", subfolder)

    print(f"Plots für Modell {n} gespeichert unter: {os.path.join(output_dir, subfolder)}")


# ============================================================
# Aufruf
# ============================================================
auswertung_modell(1)
auswertung_modell(2)