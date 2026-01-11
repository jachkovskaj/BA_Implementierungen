import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# =====================================================================
# 1. Globale Konfiguration
# =====================================================================

ROOT = Path("Ergebnisse_Teil_2")

MODEL = 2  # 1 -> Modell 1.x, 2 -> Modell 2

# - bei MODEL=1: "1.1"/"1.2"/"1.3" oder None (= alle 1.x)
# - bei MODEL=2: "2" oder None (= Modell 2)
MODEL_LABEL_FOR_PLOT = None

METRIC = "rmse"  # "rmse" oder "r2"

BEST_PER_SPLIT_SEED = 10  # beste N Trials pro (Split_Methode, Seed)

# =====================================================================
# 2. Hilfsfunktionen
# =====================================================================

def parse_study_name(study_name: str, model: int):
    """
    Extrahiert:
    - Versuchsplan (nur zum Filtern: Halton/LHS/...)
    - Modell-Label ('1.1' / '1.2' / '1.3' oder '2')
    - Submodell ('1' für 1.1, '2' für 1.2, '3' für 1.3)
    - Split-Methode ('KS'/'DUPLEX'/'SPlit'/'SPXY')
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

    m_split = re.search(r"_(KS|DUPLEX|SPlit|SPXY)_", study_name)
    split = m_split.group(1) if m_split else "Other"

    m_seed = re.search(r"seed[_-]?(\d+)", study_name)
    seed = int(m_seed.group(1)) if m_seed else None

    return plan, modell_label, sub, split, seed


def collect_all_trials_for_model(model: int, metric: str) -> pd.DataFrame:
    """
    Lädt alle COMPLETE-Trials aller passenden Studies unter ROOT und hängt
    Modell/Split/Seed an.

    Zusätzlich:
    - MODEL=1 -> nur Halton
    - MODEL=2 -> nur LHS
    """
    rows = []

    for study_dir in sorted(ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        # Modellfilter
        if model == 1 and "Modell_1" not in study_dir.name:
            continue
        if model == 2 and "Modell_2" not in study_dir.name:
            continue

        # Fixierung Versuchsplan
        if model == 1 and "_Halton_" not in study_dir.name:
            continue
        if model == 2 and "_LHS_" not in study_dir.name:
            continue

        # Study-Excel suchen
        study_file = next((f for f in study_dir.glob("*.xlsx") if "Study" in f.name), None)
        if study_file is None:
            continue

        plan, modell_label, sub, split, seed = parse_study_name(study_dir.name, model)

        df = pd.read_excel(study_file)

        df = df[df["state"] == "COMPLETE"].copy()
        if df.empty:
            continue

        # Metrik-Spalte definieren
        if metric == "rmse":
            df["metric_val"] = df["user_attrs_val_rmse"] if "user_attrs_val_rmse" in df.columns else df["value"]
        elif metric == "r2":
            df["metric_val"] = df["user_attrs_val_r2"] if "user_attrs_val_r2" in df.columns else df["value"]
        else:
            raise ValueError("METRIC muss 'rmse' oder 'r2' sein.")

        df["Modell"] = modell_label
        df["Split_Methode"] = split
        df["Seed"] = seed

        rows.append(df)

    if not rows:
        raise RuntimeError("Keine Trials gefunden – ROOT/Struktur/Modell prüfen.")

    return pd.concat(rows, ignore_index=True)


# =====================================================================
# 3. Parallel-Coordinates-Plot
# =====================================================================

def make_parallel_coordinates(df_all: pd.DataFrame,
                              model_label_for_plot,
                              metric: str,
                              best_per_split_seed: int = 50):

    # 1) Filter: nur bestimmtes Modell-Label oder alles (bei MODEL=1: alle 1.x)
    if model_label_for_plot is not None:
        df = df_all[df_all["Modell"] == model_label_for_plot].copy()
        if df.empty:
            raise ValueError(f"Keine Daten für Modell {model_label_for_plot} gefunden.")
        title_suffix = f"Modell {model_label_for_plot}"
    else:
        df = df_all.copy()
        # sinnvoller Titel abhängig vom tatsächlich geladenen Modell
        if (df["Modell"] == "2").all():
            title_suffix = "Modell 2"
        else:
            title_suffix = "Modell 1.x (alle Submodelle)"

    # 2) pro (Split_Methode, Seed) die besten N Trials auswählen
    ascending = True if metric == "rmse" else False

    df = (
        df.sort_values("metric_val", ascending=ascending)
          .groupby(["Split_Methode", "Seed"], as_index=False)
          .head(best_per_split_seed)
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

    # 4) ModelType kodieren (nur relevant, wenn mehrere Modell-Labels vorhanden sind)
    models = sorted(df["Modell"].unique())
    if len(models) > 1:
        model_mapping = {m: (i / (len(models) - 1)) for i, m in enumerate(models)}
    else:
        model_mapping = {models[0]: 0.5}

    mappings = {"Modell": model_mapping}
    df["ModelType"] = df["Modell"].map(model_mapping)

    num_cols = ["ModelType"] + base_num_cols

    # 5) kategoriale Spalten (Split statt Versuchsplan)
    cat_cols = [
        "Split_Methode",
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

    # 8) Achsennamen
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
        "Split_Methode_code": "Split",
        "ModelType": "Model",
    }
    df_scaled = df_scaled.rename(columns=rename_map)

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
        "Split",
        "Model",
    ]

    # 9) Farbskala + Colorbar (neutral, ohne gut/schlecht)
    min_v = df_scaled["metric_val"].min()
    max_v = df_scaled["metric_val"].max()

    if metric == "rmse":
        colorscale = "Teal"
        colorbar_title = "RMSE"
    else:
        colorscale = "Teal_r"
        colorbar_title = "R²"

    # 10) Plot
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
            tickvals=[min_v, max_v],
            ticktext=[f"{min_v:.3f}", f"{max_v:.3f}"],
        )
    )

    # 11) Model-Achse ticktext (nur wenn mehrere Modelle drin sind)
    if len(models) > 1:
        model_vals = [model_mapping[m] for m in models]
        for dim in fig.data[0]["dimensions"]:
            if dim.get("label") == "Model":
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
    print("Split-Methoden in df_all:")
    print(df_all["Split_Methode"].value_counts())

    fig, df_scaled, mappings = make_parallel_coordinates(
        df_all,
        model_label_for_plot=MODEL_LABEL_FOR_PLOT,
        metric=METRIC,
        best_per_split_seed=BEST_PER_SPLIT_SEED,
    )

    out_dir = Path("Ergebnisse_ParallelCoordinates_Datenaufteilung_Teil_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    label = MODEL_LABEL_FOR_PLOT if MODEL_LABEL_FOR_PLOT is not None else ("2" if MODEL == 2 else "1x_all")
    metric_label = METRIC

    fig.write_html(out_dir / f"ParallelPlot_Modell_{label}_{metric_label}_by_split_best.html")
    fig.write_image(out_dir / f"ParallelPlot_Modell_{label}_{metric_label}_by_split_best.png", scale=2)

    mappings_path = out_dir / f"Mappings_Modell_{label}_{metric_label}_by_split_best.txt"
    with open(mappings_path, "w") as f:
        for col, mapping in mappings.items():
            f.write(f"{col}:\n")
            for code, original in mapping.items():
                f.write(f"  {code} -> {original}\n")
            f.write("\n")

    print(f"Plots, Daten und Mappings gespeichert unter: {out_dir}")