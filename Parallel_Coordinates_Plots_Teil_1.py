import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px


# =====================================================================
# 1. Globale Konfiguration
# =====================================================================

ROOT = Path("Ergebnisse_Teil_1")
MODEL = 1
MODEL_LABEL_FOR_PLOT = "1.1"   # z.B. "1.3" oder None für alle 1.x
METRIC = "rmse"               # "rmse" oder "r2"
BEST_PER_PLAN_SEED = 10

# =====================================================================
# 2. Hilfsfunktionen
# =====================================================================

def parse_study_name(study_name: str, model: int):
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
    rows = []

    for study_dir in sorted(ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        if model == 1 and "Modell_1" not in study_dir.name:
            continue
        if model == 2 and "Modell_2" not in study_dir.name:
            continue

        study_file = next((f for f in study_dir.glob("*.xlsx") if "Study" in f.name), None)
        if study_file is None:
            continue

        plan, modell_label, sub, split, seed = parse_study_name(study_dir.name, model)

        df = pd.read_excel(study_file)

        df = df[df["state"] == "COMPLETE"].copy()
        if df.empty:
            continue

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

        df["Versuchsplan"] = plan
        df["Modell"] = modell_label
        df["Split_Methode"] = split
        df["Seed"] = seed

        rows.append(df)

    if not rows:
        raise RuntimeError("Keine Trials gefunden – ROOT/Struktur/Modell prüfen.")

    df_all = pd.concat(rows, ignore_index=True)
    return df_all


def frange(start, stop, step):
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(v)
        v += step
    return vals


# =====================================================================
# 3. Parallel-Coordinates-Plot
# =====================================================================

def make_parallel_coordinates(df_all: pd.DataFrame,
                              model_label_for_plot,
                              metric: str,
                              best_per_plan_seed: int = 50):

    mappings = {}

    # 1) Filter
    if model_label_for_plot is not None:
        # Wichtig: MODEL_LABEL_FOR_PLOT muss ein String sein, z.B. "1.1"
        df = df_all[df_all["Modell"] == model_label_for_plot].copy()
        if df.empty:
            raise ValueError(f"Keine Daten für Modell {model_label_for_plot} gefunden.")
        title_suffix = f"Modell {model_label_for_plot}"
    else:
        df = df_all.copy()
        if df["Modell"].nunique() == 1:
            title_suffix = f"Modell {df['Modell'].iloc[0]}"
        else:
            title_suffix = "Modell 1.x (alle Submodelle)"

    # 2) beste N pro (Plan, Seed, ggf. Modell)
    ascending = (metric == "rmse")

    group_cols = ["Versuchsplan", "Seed"]
    # Modell nur in die Gruppierung nehmen, wenn wir mehrere Modelle gleichzeitig betrachten
    if MODEL == 1 and model_label_for_plot is None:
        group_cols.append("Modell")

    df = (
        df.sort_values("metric_val", ascending=ascending)
          .groupby(group_cols, as_index=False)
          .head(best_per_plan_seed)
          .reset_index(drop=True)
    )

    base_num_cols = [
        "params_Batch_Size",
        "params_Learning_Rate",
        "params_Weight_Decay",
        "params_n_Layers",
        "params_n_units_l0",
        "params_n_units_l1",
        "params_n_units_l2",
    ]

    # --------------------------------------------------------
    # Sollen wir eine Model-Achse verwenden?
    # -> nur bei MODEL == 1 UND wenn mehrere Modelle gleichzeitig geplottet werden
    # --------------------------------------------------------
    use_model_axis = (MODEL == 1 and model_label_for_plot is None and df["Modell"].nunique() > 1)

    if use_model_axis:
        models = sorted(df["Modell"].unique())
        if len(models) > 1:
            model_mapping = {m: (i / (len(models) - 1)) for i, m in enumerate(models)}
        else:
            model_mapping = {models[0]: 0.5}

        mappings["Modell"] = model_mapping
        df["ModelType"] = df["Modell"].map(model_mapping)
        num_cols = ["ModelType"] + base_num_cols
    else:
        # kein ModelType, keine Model-Achse
        model_mapping = {}
        models = []
        num_cols = base_num_cols

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

    # Kategorie-Mappings befüllen
    for col in cat_cols:
        cat = df_plot[col].astype("category")
        df_plot[col + "_code"] = cat.cat.codes
        mappings[col] = dict(enumerate(cat.cat.categories))

    cols_for_plot = num_cols + [c + "_code" for c in cat_cols]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_plot[cols_for_plot]),
        columns=cols_for_plot,
    )
    df_scaled["metric_val"] = df_plot["metric_val"].values

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
        "Versuchsplan",
    ]
    if use_model_axis:
        dimensions.append("Model")

    # 9) Farbskala + Beschriftung + Colorbar-Ticks
    min_v = df_scaled["metric_val"].min()
    max_v = df_scaled["metric_val"].max()

    if metric == "rmse":
        colorscale = "Teal"
        colorbar_title = "RMSE"
        step = 0.5 if max_v < 5 else 5.0
    else:
        colorscale = "Teal_r"
        colorbar_title = "R²"
        step = 0.1

    start = (min_v // step) * step
    base_ticks = [round(v, 2) for v in frange(start, max_v, step) if v >= min_v - 1e-9]

    all_ticks = sorted(set(base_ticks + [min_v, max_v]))
    ticktext = [f"{v:.2f}" for v in all_ticks]

    fig = px.parallel_coordinates(
        df_scaled,
        dimensions=dimensions,
        color="metric_val",
        color_continuous_scale=colorscale,
        title=f"Parallel Coordinates – {title_suffix}",
    )

    idx_min = all_ticks.index(min_v)
    idx_max = all_ticks.index(max_v)

    if metric == "rmse":
        ticktext[idx_min] = f"{min_v:.2f}  (gut)"
        ticktext[idx_max] = f"{max_v:.2f}  (schlecht)"
    else:
        ticktext[idx_min] = f"{min_v:.2f}  (schlecht)"
        ticktext[idx_max] = f"{max_v:.2f}  (gut)"

    fig.update_coloraxes(
        colorbar_title=colorbar_title,
        colorbar=dict(
            tickvals=all_ticks,
            ticktext=ticktext,
            ticks="outside",
        )
    )

    # Model-Achse nur beschriften, wenn sie überhaupt existiert
    if use_model_axis and model_mapping:
        model_vals = [model_mapping[m] for m in models]
        for dim in fig.data[0]["dimensions"]:
            if dim["label"] == "Model":
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

    out_dir = Path("Ergebnisse_ParallelCoordinates")
    out_dir.mkdir(parents=True, exist_ok=True)

    label = MODEL_LABEL_FOR_PLOT if MODEL_LABEL_FOR_PLOT is not None else "1x_all"
    metric_label = METRIC

    fig.write_html(out_dir / f"ParallelPlot_Modell_{label}_{metric_label}.html")
    fig.write_image(out_dir / f"ParallelPlot_Modell_{label}_{metric_label}.png", scale=2)

    mappings_path = out_dir / f"Mappings_Modell_{label}_{metric_label}.txt"
    with open(mappings_path, "w") as f:
        for col, mapping in mappings.items():
            f.write(f"{col}:\n")
            for code, original in mapping.items():
                f.write(f"  {code} -> {original}\n")
            f.write("\n")

    print(f"Plots, Daten und Mappings gespeichert unter: {out_dir}")