import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# =====================================================================
# 1. Globale Konfiguration
# =====================================================================

# Basisordner mit deinen Study-Unterordnern
ROOT = Path("Ergebnisse_Teil_1")

# Welches Hauptmodell auswerten? 1 -> Modell 1.x (1.1, 1.2, 1.3), 2 -> Modell 2
MODEL = 2

# Welches Submodell soll geplottet werden?
# - "1.1", "1.2", "1.3"  -> nur dieses Submodell
# - "2"                  -> Modell 2
# - None                 -> alle 1.x gemeinsam (nur sinnvoll bei MODEL = 1)
MODEL_LABEL_FOR_PLOT = None   # z.B. "1.3" oder None für alle 1.x

# Welche Metrik soll für die Farbe verwendet werden?
# "rmse" -> kleiner ist besser
# "r2"   -> größer ist besser
METRIC = "rmse"

# Wie viele der besten Trials pro (Versuchsplan, Seed) im Plot anzeigen?
BEST_PER_PLAN_SEED = 10

# =====================================================================
# 2. Hilfsfunktionen
# =====================================================================

def parse_study_name(study_name: str, model: int):
    """
    Analysiert den Ordner- bzw. Dateinamen und extrahiert:
    - Versuchsplan (Halton, LHS, Sobol, Taguchi)
    - Modell-Label (z.B. '1.1' oder '2')
    - Submodell (z.B. '1' für 1.1, 2 für 1.2, ...)
    - Split-Methode (z.B. 'KS_Holdout')
    - Seed (int)
    """
    m_plan = re.search(r"(Halton|LHS|Sobol|Taguchi)", study_name, re.I)
    plan = m_plan.group(1) if m_plan else None

    if model == 1:
        m_sub = re.search(r"Modell_1[._](\d)", study_name)
        sub = m_sub.group(1) if m_sub else None
        modell_label = f"1.{sub}" if sub else "1"
    else:
        sub = None
        modell_label = "2"

    m_split = re.search(r"Modell_\d(?:\.\d)?_(.*?)_seed", study_name)
    split = m_split.group(1) if m_split else None

    m_seed = re.search(r"seed[_-]?(\d+)", study_name)
    seed = int(m_seed.group(1)) if m_seed else None

    return plan, modell_label, sub, split, seed

def collect_all_trials_for_model(model: int, metric: str) -> pd.DataFrame:
    """
    Lädt für MODEL (1 oder 2) alle COMPLETE-Trials aus allen Study-Ordnern
    unter ROOT und hängt Plan/Modell/Split/Seed als Spalten an.

    metric:
      "rmse" -> erwartet user_attrs_val_rmse oder nutzt value als RMSE
      "r2"   -> erwartet user_attrs_val_r2   oder nutzt value als R²
    """
    rows = []

    for study_dir in sorted(ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        # nur passende Modellordner berücksichtigen
        if model == 1 and "Modell_1" not in study_dir.name:
            continue
        if model == 2 and "Modell_2" not in study_dir.name:
            continue

        # Study-Excel im Ordner suchen
        study_file = next((f for f in study_dir.glob("*.xlsx") if "Study" in f.name), None)
        if study_file is None:
            continue

        plan, modell_label, sub, split, seed = parse_study_name(study_dir.name, model)

        df = pd.read_excel(study_file)

        # nur COMPLETE-Trials
        df = df[df["state"] == "COMPLETE"].copy()
        if df.empty:
            continue

        # Metrik-Spalte definieren
        if metric == "rmse":
            if "user_attrs_val_rmse" in df.columns:
                df["metric_val"] = df["user_attrs_val_rmse"]
            else:
                df["metric_val"] = df["value"]
        elif metric == "r2":
            if "user_attrs_val_r2" in df.columns:
                df["metric_val"] = df["user_attrs_val_r2"]
            else:
                df["metric_val"] = df["value"]
        else:
            raise ValueError("METRIC muss 'rmse' oder 'r2' sein.")

        # Meta-Infos anhängen
        df["Versuchsplan"] = plan
        df["Modell"] = modell_label      # wichtig für Model-Type-Achse
        df["Split_Methode"] = split
        df["Seed"] = seed

        rows.append(df)

    if not rows:
        raise RuntimeError("Keine Trials gefunden – ROOT/Struktur/Modell prüfen.")

    df_all = pd.concat(rows, ignore_index=True)
    return df_all

# =====================================================================
# 3. Parallel-Coordinates-Plot
# =====================================================================

def make_parallel_coordinates(df_all: pd.DataFrame,
                              model_label_for_plot,
                              metric: str,
                              best_per_plan_seed: int = 50):

    # 1) Filter: nur bestimmtes Submodell oder alle?
    if model_label_for_plot is not None:
        # z.B. nur "1.3"
        df = df_all[df_all["Modell"] == model_label_for_plot].copy()
        if df.empty:
            raise ValueError(f"Keine Daten für Modell {model_label_for_plot} gefunden.")
        title_suffix = f"Modell {model_label_for_plot}"
    else:
        # alle 1.x gemeinsam
        df = df_all.copy()
        title_suffix = "Modell 1.x (alle Submodelle)"

    # 2) pro (Versuchsplan, Seed) die besten N Trials auswählen
    if metric == "rmse":
        ascending = True   # kleiner ist besser
    else:  # "r2"
        ascending = False  # größer ist besser

    df = (
        df.sort_values("metric_val", ascending=ascending)
          .groupby(["Versuchsplan", "Seed"], as_index=False)
          .head(best_per_plan_seed)
          .reset_index(drop=True)
    )

    # 3) numerische Hyperparameter
    base_num_cols = [
        "params_Batch_Size",
        "params_Learning_Rate",
        "params_Weight_Decay",
        "params_n_Layers",
        "params_n_units_l0",
        "params_n_units_l1",
        "params_n_units_l2",
    ]

    # 4) Model Type explizit kodieren: 1.1 -> 0.0, 1.2 -> 0.5, 1.3 -> 1.0
    models = sorted(df["Modell"].unique())  # z.B. ['1.1', '1.2', '1.3']
    if len(models) > 1:
        model_mapping = {
            m: (i / (len(models) - 1)) for i, m in enumerate(models)
        }  # bei 3 Modellen: 0.0, 0.5, 1.0
    else:
        # falls nur ein Modell vorhanden ist
        model_mapping = {models[0]: 0.5}

    mappings = {
        "Modell": model_mapping
    }

    df["ModelType"] = df["Modell"].map(model_mapping)
    num_cols = ["ModelType"] + base_num_cols  # ModelType als erste numerische Achse

    # 5) weitere kategoriale Spalten
    cat_cols = [
        "Versuchsplan",
        "params_act_func_l0",
        "params_act_func_l1",
        "params_act_func_l2",
    ]

    missing = [c for c in num_cols + cat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Spalten fehlen: {missing}")

    df_plot = df[num_cols + cat_cols + ["metric_val"]].copy()

    # 6) Kategoriale Variablen → Codes + Mapping speichern
    for col in cat_cols:
        cat = df_plot[col].astype("category")
        df_plot[col + "_code"] = cat.cat.codes
        mappings[col] = dict(enumerate(cat.cat.categories))

    cols_for_plot = num_cols + [c + "_code" for c in cat_cols]

    # 7) Skalieren auf [0, 1]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_plot[cols_for_plot]),
        columns=cols_for_plot,
    )
    df_scaled["metric_val"] = df_plot["metric_val"].values

    # 8) Schöne Achsennamen
    rename_map = {
        "params_Batch_Size": "Batch Size",
        "params_Learning_Rate": "Learning Rate",
        "params_Weight_Decay": "Weight Decay",
        "params_n_Layers": "Hidden Layers",
        "params_n_units_l0": "Neurons L0",
        "params_n_units_l1": "Neurons L1",
        "params_n_units_l2": "Neurons L2",
        "params_act_func_l0_code": "Activation L0",
        "params_act_func_l1_code": "Activation L1",
        "params_act_func_l2_code": "Activation L2",
        "Versuchsplan_code": "Versuchsplan",
        "ModelType": "Model"
    }

    df_scaled = df_scaled.rename(columns=rename_map)

    # gewünschte Reihenfolge der Achsen
    dimensions = [
        "Batch Size",
        "Learning Rate",
        "Weight Decay",
        "Hidden Layers",
        "Neurons L0",
        "Neurons L1",
        "Neurons L2",
        "Activation L0",
        "Activation L1",
        "Activation L2",
        "Versuchsplan",
        "Model"
    ]

    # 9) Farbskala + Beschriftung
    min_v = df_scaled["metric_val"].min()
    max_v = df_scaled["metric_val"].max()

    if metric == "rmse":
        colorscale = "Teal"      # farbenblind-sicher
        colorbar_title = "RMSE"
        ticktext = [f"{min_v:.3f}  (gut)", f"{max_v:.3f}  (schlecht)"]
    else:
        colorscale = "Teal_r"    # invertiert, damit hohes R² dunkel ist
        colorbar_title = "R²"
        ticktext = [f"{min_v:.3f}  (schlecht)", f"{max_v:.3f}  (gut)"]

    tickvals = [min_v, max_v]

    # 10) Plot erstellen
    fig = px.parallel_coordinates(
        df_scaled,
        dimensions=dimensions,
        color="metric_val",
        color_continuous_scale=colorscale,
        title=f"Parallel Coordinates – {title_suffix}",
    )

    fig.update_coloraxes(
        colorbar_title=colorbar_title,
        colorbar=dict(
            tickvals=tickvals,
            ticktext=ticktext,
        )
    )

    # 11) Ticks für Model-Type-Achse sauber setzen (0 / 0.5 / 1 → 1.1 / 1.2 / 1.3)
    if len(models) > 1:
        model_vals = [model_mapping[m] for m in models]
        for dim in fig.data[0]["dimensions"]:
            if dim["label"] == "Model Type":
                dim["tickvals"] = model_vals
                dim["ticktext"] = models
                break

    fig.show()

    return fig, df_scaled, mappings

# =====================================================================
# 4. Main
# =====================================================================

if __name__ == "__main__":
    print(f"Sammle Trials für Modell {MODEL} aus {ROOT} ...")
    df_all = collect_all_trials_for_model(MODEL, METRIC)
    print(f"{len(df_all)} COMPLETE-Trials geladen.")
    print("Modell-Typen in df_all:")
    print(df_all["Modell"].value_counts())

    fig, df_scaled, mappings = make_parallel_coordinates(
        df_all,
        model_label_for_plot=MODEL_LABEL_FOR_PLOT,
        metric=METRIC,
        best_per_plan_seed=BEST_PER_PLAN_SEED,
    )

    # Zielordner
    out_dir = Path("Ergebnisse_ParallelCoordinates")
    out_dir.mkdir(parents=True, exist_ok=True)

    label = MODEL_LABEL_FOR_PLOT if MODEL_LABEL_FOR_PLOT is not None else "1x_all"
    metric_label = METRIC

    # Plot speichern
    fig.write_html(out_dir / f"ParallelPlot_Modell_{label}_{metric_label}_overall_best.html")
    fig.write_image(out_dir / f"ParallelPlot_Modell_{label}_{metric_label}_overall_best.png", scale=2)

    # Mappings als TXT speichern
    mappings_path = out_dir / f"Mappings_Modell_{label}_{metric_label}_overall_best.txt"
    with open(mappings_path, "w") as f:
        for col, mapping in mappings.items():
            f.write(f"{col}:\n")
            for code, original in mapping.items():
                f.write(f"  {code} -> {original}\n")
            f.write("\n")

    print(f"Plots, Daten und Mappings gespeichert unter: {out_dir}")