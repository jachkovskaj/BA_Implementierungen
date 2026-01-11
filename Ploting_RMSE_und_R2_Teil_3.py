import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def _load_best_metrics_for_reduction(base_dir,
                                     reduction_name,   # "CBIR" oder "RegMix"
                                     model_nummer,
                                     submodel_nummer=None,
                                     seeds=(0, 42, 999),
                                     fixed_plan=None):
    """
    Fix:
      - Modell 1: Plan=Halton, Split=SPXY
      - Modell 2: Plan=LHS,   Split=KS

    Besonderheit CBIR:
      - kann beliebig oft als ..._CONF_80/85/90/95... vorkommen → wir suchen per Glob.
    """
    if fixed_plan is None:
        raise ValueError("fixed_plan muss gesetzt sein (z.B. 'Halton' oder 'LHS').")

    fixed_split = "SPXY" if model_nummer == 1 else "KS"

    metrics_list = []

    for seed in seeds:
        # ---------------------------
        # Study-Verzeichnis finden
        # ---------------------------
        if model_nummer == 1:
            if submodel_nummer is None:
                raise ValueError("Für Modell 1 muss submodel_nummer (1, 2 oder 3) angegeben werden.")

            if reduction_name.upper() == "CBIR":
                # Modell 1, CBIR mit CONF (beliebig)
                patt = (
                    f"Study_15_10_2025_{fixed_plan}_Modell_{model_nummer}.{submodel_nummer}_"
                    f"{fixed_plan}_{fixed_split}_CBIR_Holdout_seed_{seed}_CONF_*"
                )
                matches = sorted((Path(base_dir)).glob(patt))
                if not matches:
                    print(f"[Hinweis] Keine CBIR-CONF Studies gefunden für Seed {seed}: {patt}")
                    continue
                # nimm einfach die erste (oder die letzte) – falls mehrere existieren
                study_dir = matches[0]
                study_name = study_dir.name
            else:
                # Modell 1, RegMix (ohne CONF)
                study_name = (
                    f"Study_15_10_2025_{fixed_plan}_Modell_{model_nummer}.{submodel_nummer}_"
                    f"{fixed_plan}_{fixed_split}_{reduction_name}_Holdout_seed_{seed}"
                )
                study_dir = Path(base_dir) / study_name

        else:
            # ---------------------------
            # ✅ Modell 2: Prefix MUSS LHS sein
            # ---------------------------
            if reduction_name.upper() == "CBIR":
                patt = (
                    f"Study_15_10_2025_{fixed_plan}_Modell_{model_nummer}_"
                    f"{fixed_plan}_{fixed_split}_CBIR_Holdout_seed_{seed}_CONF_*"
                )
                matches = sorted((Path(base_dir)).glob(patt))
                if not matches:
                    print(f"[Hinweis] Keine CBIR-CONF Studies gefunden für Seed {seed}: {patt}")
                    continue
                study_dir = matches[0]
                study_name = study_dir.name
            else:
                study_name = (
                    f"Study_15_10_2025_{fixed_plan}_Modell_{model_nummer}_"
                    f"{fixed_plan}_{fixed_split}_{reduction_name}_Holdout_seed_{seed}"
                )
                study_dir = Path(base_dir) / study_name

        # ---------------------------
        # Study-Excel laden
        # ---------------------------
        study_file = study_dir / f"{study_name}.xlsx"
        if not study_file.exists():
            print(f"[Hinweis] Study-Datei fehlt: {study_file}")
            continue

        try:
            df_study = pd.read_excel(study_file, engine="openpyxl")
        except Exception as e:
            print(f"[Fehler] Konnte Study-Datei {study_file} nicht lesen: {e}")
            continue

        if "state" not in df_study.columns or "value" not in df_study.columns:
            print(f"[Hinweis] Study-Datei {study_file} hat kein 'state'/'value' → überspringe.")
            continue

        df_complete = df_study[df_study["state"] == "COMPLETE"]
        if df_complete.empty:
            print(f"[Hinweis] Study-Datei {study_file}: keine COMPLETE-Trials.")
            continue

        best_row = df_complete.loc[df_complete["value"].idxmin()]
        trial_num = int(best_row["number"]) if "number" in df_complete.columns else int(best_row.name)

        # ---------------------------
        # metrics_{trial}.csv laden
        # ---------------------------
        metrics_file = study_dir / "metrics" / f"metrics_{trial_num}.csv"
        if not metrics_file.exists():
            print(f"[Hinweis] Metrics-Datei fehlt: {metrics_file}")
            continue

        try:
            df_metrics = pd.read_csv(metrics_file)
        except Exception as e:
            print(f"[Fehler] Konnte Metrics-Datei {metrics_file} nicht lesen: {e}")
            continue

        lower_map = {c.lower(): c for c in df_metrics.columns}
        needed = ["dataset", "rmse", "r2"]
        if not all(col in lower_map for col in needed):
            print(f"[Hinweis] Metrics-Datei {metrics_file} ohne (dataset/rmse/r2) → Spalten: {list(df_metrics.columns)}")
            continue

        df_metrics = df_metrics.rename(columns={
            lower_map["dataset"]: "Dataset",
            lower_map["rmse"]: "RMSE",
            lower_map["r2"]: "R2",
        }).set_index("Dataset")

        metrics_list.append(df_metrics)

    if not metrics_list:
        label = f"{model_nummer}" if submodel_nummer is None else f"{model_nummer}.{submodel_nummer}"
        raise RuntimeError(f"Keine gültigen Metrics für {reduction_name}, Modell {label} gefunden.")

    df_all = pd.concat(metrics_list, axis=0)
    df_mean = df_all.groupby(df_all.index).mean().reindex(["Train", "Validation", "Test"])
    df_std  = df_all.groupby(df_all.index).std(ddof=1).reindex(["Train", "Validation", "Test"])
    return df_mean, df_std

def plots(model_nummer,
          submodel_nummer=None,
          seeds=(0, 42, 999),
          base_dir="Ergebnisse_Teil_3"):

    # Konsistenz prüfen + Plan fixieren
    if model_nummer == 1:
        if submodel_nummer is None:
            raise ValueError("Für Modell 1 muss submodel_nummer (1, 2 oder 3) angegeben werden.")
        fixed_plan = "Halton"
    elif model_nummer == 2:
        submodel_nummer = None
        fixed_plan = "LHS"
    else:
        raise ValueError("model_nummer muss 1 oder 2 sein.")

    try:
        plt.style.use(["IKV.mplstyle"])
    except Exception:
        pass

    reductions = ["CBIR", "RegMix"]

    data_rmse_mean = {}
    data_rmse_std  = {}
    data_r2_mean   = {}
    data_r2_std    = {}

    for red in reductions:
        df_mean, df_std = _load_best_metrics_for_reduction(
            base_dir=base_dir,
            reduction_name=red,
            model_nummer=model_nummer,
            submodel_nummer=submodel_nummer,
            seeds=seeds,
            fixed_plan=fixed_plan
        )

        data_rmse_mean[red] = (
            df_mean.loc["Train", "RMSE"],
            df_mean.loc["Validation", "RMSE"],
            df_mean.loc["Test", "RMSE"],
        )
        data_rmse_std[red] = (
            df_std.loc["Train", "RMSE"],
            df_std.loc["Validation", "RMSE"],
            df_std.loc["Test", "RMSE"],
        )

        data_r2_mean[red] = (
            df_mean.loc["Train", "R2"],
            df_mean.loc["Validation", "R2"],
            df_mean.loc["Test", "R2"],
        )
        data_r2_std[red] = (
            df_std.loc["Train", "R2"],
            df_std.loc["Validation", "R2"],
            df_std.loc["Test", "R2"],
        )

    # Arrays bauen
    x = np.arange(len(reductions))
    width = 0.25

    rmse_train_vals = np.array([data_rmse_mean[s][0] for s in reductions])
    rmse_val_vals   = np.array([data_rmse_mean[s][1] for s in reductions])
    rmse_test_vals  = np.array([data_rmse_mean[s][2] for s in reductions])

    rmse_train_std  = np.nan_to_num(np.array([data_rmse_std[s][0] for s in reductions]))
    rmse_val_std    = np.nan_to_num(np.array([data_rmse_std[s][1] for s in reductions]))
    rmse_test_std   = np.nan_to_num(np.array([data_rmse_std[s][2] for s in reductions]))

    r2_train_vals = np.array([data_r2_mean[s][0] for s in reductions])
    r2_val_vals   = np.array([data_r2_mean[s][1] for s in reductions])
    r2_test_vals  = np.array([data_r2_mean[s][2] for s in reductions])

    r2_train_std  = np.nan_to_num(np.array([data_r2_std[s][0] for s in reductions]))
    r2_val_std    = np.nan_to_num(np.array([data_r2_std[s][1] for s in reductions]))
    r2_test_std   = np.nan_to_num(np.array([data_r2_std[s][2] for s in reductions]))

    modell_label = f"{model_nummer}" if submodel_nummer is None else f"{model_nummer}.{submodel_nummer}"
    out_dir = f"Ergebnisplots_Visualization_Metriken_Datenreduktion/Plots_Vergleich_{fixed_plan}_Modell_{modell_label}"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) RMSE Train / Val / Test ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, rmse_train_vals, width=width, yerr=rmse_train_std, capsize=4, label="Train")
    ax.bar(x,         rmse_val_vals,   width=width, yerr=rmse_val_std,   capsize=4, label="Validation")
    ax.bar(x + width, rmse_test_vals,  width=width, yerr=rmse_test_std,  capsize=4, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(reductions, rotation=90)
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE von Train / Val / Test – {fixed_plan}, Modell {modell_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fixed_plan}_Modell_{modell_label}_Plot_1_RMSE_Train_Val_Test.png"))
    plt.close(fig)

    # ---------- 2) R² Train / Val / Test ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, r2_train_vals, width=width, yerr=r2_train_std, capsize=4, label="Train")
    ax.bar(x,         r2_val_vals,   width=width, yerr=r2_val_std,   capsize=4, label="Validation")
    ax.bar(x + width, r2_test_vals,  width=width, yerr=r2_test_std,  capsize=4, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(reductions, rotation=90)
    ax.set_ylabel("R²")
    ax.set_title(f"R² von Train / Val / Test – {fixed_plan}, Modell {modell_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fixed_plan}_Modell_{modell_label}_Plot_2_R2_Train_Val_Test.png"))
    plt.close(fig)

    # ---------- 3) Kombi-Plot: RMSE & R² (Train/Val/Test) ----------
    fig, ax_rmse = plt.subplots(figsize=(10, 5))
    ax_r2 = ax_rmse.twinx()
    w = 0.10

    c_rmse_train = "#95BB20"
    c_rmse_val = "#00354E"
    c_rmse_test = "#717E86"
    c_r2_train = "#BA274A"
    c_r2_val = "#0098AD"
    c_r2_test = "#F7A600"

    for i in range(len(reductions)):
        x0 = x[i]

        ax_rmse.bar(x0 - 2*w, rmse_train_vals[i], width=w, yerr=rmse_train_std[i], capsize=3,
                    color=c_rmse_train, label="RMSE Train" if i == 0 else "")
        ax_rmse.bar(x0 - 1*w, rmse_val_vals[i],   width=w, yerr=rmse_val_std[i],   capsize=3,
                    color=c_rmse_val,   label="RMSE Validation" if i == 0 else "")
        ax_rmse.bar(x0,        rmse_test_vals[i], width=w, yerr=rmse_test_std[i],  capsize=3,
                    color=c_rmse_test,  label="RMSE Test" if i == 0 else "")

        ax_r2.bar(x0 + 1*w, r2_train_vals[i], width=w, yerr=r2_train_std[i], capsize=3,
                  color=c_r2_train, label="R² Train" if i == 0 else "")
        ax_r2.bar(x0 + 2*w, r2_val_vals[i],   width=w, yerr=r2_val_std[i],   capsize=3,
                  color=c_r2_val,   label="R² Validation" if i == 0 else "")
        ax_r2.bar(x0 + 3*w, r2_test_vals[i],  width=w, yerr=r2_test_std[i],  capsize=3,
                  color=c_r2_test,  label="R² Test" if i == 0 else "")

    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(reductions, rotation=90)
    ax_rmse.set_ylabel("RMSE")
    ax_r2.set_ylabel("R²")
    ax_rmse.set_title(f"RMSE & R² (Train / Validation / Test) – {fixed_plan}, Modell {modell_label}")
    ax_rmse.grid(axis="y", linestyle="--", alpha=0.7)

    h1, l1 = ax_rmse.get_legend_handles_labels()
    h2, l2 = ax_r2.get_legend_handles_labels()
    ax_rmse.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fixed_plan}_Modell_{modell_label}_Plot_3_RMSE_R2_Train_Val_Test.png"))
    plt.close(fig)

    # ---------- 4) RMSE nur Test ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, rmse_test_vals, yerr=rmse_test_std, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(reductions, rotation=90)
    ax.set_ylabel("RMSE Test")
    ax.set_title(f"RMSE (Test) – {fixed_plan}, Modell {modell_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fixed_plan}_Modell_{modell_label}_Plot_4_RMSE_nur_Test.png"))
    plt.close(fig)

    # ---------- 5) R² nur Test ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, r2_test_vals, yerr=r2_test_std, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(reductions, rotation=90)
    ax.set_ylabel("R² Test")
    ax.set_title(f"R² (Test) – {fixed_plan}, Modell {modell_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fixed_plan}_Modell_{modell_label}_Plot_5_R2_nur_Test.png"))
    plt.close(fig)

    # ---------- 6) Kombi-Plot: RMSE & R² nur Test ----------
    fig, ax_rmse_test = plt.subplots(figsize=(6, 5))
    ax_r2_test = ax_rmse_test.twinx()
    w = 0.30

    c_rmse_test = "#95BB20"
    c_r2_test   = "#00354E"

    ax_rmse_test.bar(x - w/2, rmse_test_vals, width=w,
                     yerr=rmse_test_std, capsize=4,
                     color=c_rmse_test, label="RMSE Test")
    ax_r2_test.bar(x + w/2, r2_test_vals, width=w,
                   yerr=r2_test_std, capsize=4,
                   color=c_r2_test, label="R² Test")

    ax_rmse_test.set_xticks(x)
    ax_rmse_test.set_xticklabels(reductions, rotation=90)
    ax_rmse_test.set_ylabel("RMSE Test")
    ax_r2_test.set_ylabel("R² Test")
    ax_rmse_test.set_title(f"RMSE & R² (Test) – {fixed_plan}, Modell {modell_label}")
    ax_rmse_test.grid(axis="y", linestyle="--", alpha=0.7)

    h1, l1 = ax_rmse_test.get_legend_handles_labels()
    h2, l2 = ax_r2_test.get_legend_handles_labels()
    ax_rmse_test.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fixed_plan}_Modell_{modell_label}_Plot_6_RMSE_R2_Test.png"))
    plt.close(fig)

    print(f"[OK] Alle Vergleichsplots für {fixed_plan}, Modell {modell_label} in {out_dir} gespeichert.")


# =========================
# Aufrufe
# =========================
plots(model_nummer=1, submodel_nummer=1)
plots(model_nummer=1, submodel_nummer=2)
plots(model_nummer=1, submodel_nummer=3)

plots(model_nummer=2)