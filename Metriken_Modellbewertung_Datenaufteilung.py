# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# KONFIG
# =========================
BASE_DIR = Path("Ergebnisse_Summary_Reeval_Datenaufteilung")  # Root, in dem Reeval_* Ordner liegen
OUT_DIR = BASE_DIR

BEST_PER_SPLIT_AND_SEED = True

# Study-Name:
# Modell 1: Study_15_10_2025_Halton_Modell_1.<sub>_<split>_Holdout_seed_<seed>
# Modell 2: Study_15_10_2025_LHS_Modell_2         _<split>_Holdout_seed_<seed>
#      oder Study_15_10_2025_LHS_Modell_2.<sub>_<split>_Holdout_seed_<seed>
STUDY_RX = re.compile(
    r"^Study_15_10_2025_"
    r"(?P<plan>Halton|LHS)_"
    r"Modell_(?P<model>[12])"
    r"(?:\.(?P<sub>[123]))?"          # sub optional (für Modell 2 erlaubt)
    r"_(?P<split>KS|DUPLEX|SPlit|SPXY)_Holdout_seed_(?P<seed>0|42|999)$",
    re.I
)

def reeval_root_for_model(model: int) -> Path:
    if model == 1:
        return BASE_DIR / "Reeval_Halton_Modell_1"
    elif model == 2:
        return BASE_DIR / "Reeval_LHS_Modell_2"
    else:
        raise ValueError("model muss 1 oder 2 sein")

def parse_study_name(study_name: str):
    m = STUDY_RX.match(study_name)
    if not m:
        return None

    model = int(m.group("model"))
    sub = m.group("sub")  # kann None sein

    # Regeln: Modell 1 braucht sub zwingend, Modell 2 darf ohne sub
    if model == 1 and sub is None:
        return None

    return {
        "plan": m.group("plan"),
        "model": model,
        "sub": sub,  # None möglich
        "split": m.group("split"),
        "seed": int(m.group("seed")),
    }

def extract_split_seed_from_path(p: Path):
    split = None
    seed = None

    for part in p.parts:
        m = re.match(r"Split_(KS|DUPLEX|SPlit|SPXY)$", part, re.I)
        if m:
            split = m.group(1)
            break

    for part in p.parts:
        m = re.match(r"seed_(0|42|999)$", part, re.I)
        if m:
            seed = int(m.group(1))
            break

    return split, seed


# =========================
# LOADER: Reevaluation Summaries
# =========================
def load_reeval_summaries(model: int) -> pd.DataFrame:
    """
    - sucht NUR im Reeval-Ordner des gewünschten Modells
    - sammelt alle summary_*.xlsx rekursiv
    - ergänzt plan/model/sub/split/seed robust
    """
    root = reeval_root_for_model(model)
    files = sorted(root.rglob("summary_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"Keine summary_*.xlsx unter {root} gefunden.")

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f)
        except Exception:
            continue

        if df is None or df.empty:
            continue

        required = {"study", "trial", "rmse_test", "r2_test"}
        if not required.issubset(df.columns):
            continue

        df = df.copy()
        df["summary_file"] = str(f)

        # split/seed aus Pfad (wenn nicht in Tabelle oder leer)
        split_from_path, seed_from_path = extract_split_seed_from_path(f)

        if "split" not in df.columns:
            df["split"] = split_from_path
        else:
            df["split"] = df["split"].where(df["split"].notna(), split_from_path)

        if "seed" not in df.columns:
            df["seed"] = seed_from_path
        else:
            df["seed"] = df["seed"].where(df["seed"].notna(), seed_from_path)

        # plan/model/sub aus study-name
        meta = df["study"].apply(parse_study_name)

        # Spalten anlegen falls fehlen
        # Spalten sauber typisieren, damit wir Strings setzen dürfen (keine float-Spalten)
        if "plan" not in df.columns:
            df["plan"] = pd.Series([None] * len(df), dtype="object")
        else:
            df["plan"] = df["plan"].astype("object")

        if "sub" not in df.columns:
            df["sub"] = pd.Series([None] * len(df), dtype="object")
        else:
            df["sub"] = df["sub"].astype("object")

        # model am besten direkt numerisch halten
        if "model" not in df.columns:
            df["model"] = pd.Series([np.nan] * len(df), dtype="float64")
        # (model bleibt später sowieso numeric via to_numeric)

        # füllen, wenn leer
        df.loc[df["plan"].isna(), "plan"] = meta.apply(lambda x: x["plan"] if x else np.nan)
        df.loc[df["model"].isna(), "model"] = meta.apply(lambda x: x["model"] if x else np.nan)
        df.loc[df["sub"].isna(), "sub"] = meta.apply(lambda x: x["sub"] if x else np.nan)

        # split/seed ggf. auch aus study-name (falls immer noch leer)
        df.loc[df["split"].isna(), "split"] = meta.apply(lambda x: x["split"] if x else np.nan)
        df.loc[df["seed"].isna(), "seed"] = meta.apply(lambda x: x["seed"] if x else np.nan)

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"Es wurden summary-Dateien unter {root} gefunden, aber keine hatte die erwarteten Spalten "
            f"(study/trial/rmse_test/r2_test)."
        )

    out = pd.concat(dfs, ignore_index=True)

    # numerische Konvertierung
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out["rmse_test"] = pd.to_numeric(out["rmse_test"], errors="coerce")
    out["r2_test"] = pd.to_numeric(out["r2_test"], errors="coerce")
    out["model"] = pd.to_numeric(out["model"], errors="coerce")

    # Zeilen bereinigen
    out = out.dropna(subset=["study", "trial", "rmse_test", "r2_test", "model"]).reset_index(drop=True)

    # jetzt sicher filtern
    out = out[out["model"].astype(int) == int(model)].reset_index(drop=True)

    if out.empty:
        found_models = sorted(pd.unique(pd.to_numeric(pd.concat(dfs, ignore_index=True)["model"], errors="coerce").dropna()))
        raise FileNotFoundError(
            f"Summary-Dateien unter {root} gefunden, aber keine Zeilen für Modell {model}.\n"
            f"Gefundene model-Werte: {found_models}\n"
            f"Hinweis: Prüfe Study-Namen von Modell 2 (Modell_2 vs Modell_2.x)."
        )

    return out


# =========================
# BESTE TRIALS AUS REEVAL
# =========================
def best_per_study(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["study"]

    if BEST_PER_SPLIT_AND_SEED:
        if "split" in df.columns:
            group_cols.append("split")
        if "seed" in df.columns:
            group_cols.append("seed")

    idx = df.groupby(group_cols)["rmse_test"].idxmin()
    best = df.loc[idx].copy()

    best = best.rename(columns={
        "rmse_test": "bester_rmse_test",
        "r2_test": "bester_r2_test",
    })
    return best.reset_index(drop=True)


# =========================
# METRIKEN
# =========================
def stability_by_group(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col)[metric]
        .agg(["mean", "std", "var", "min", "max", "count"])
        .rename(columns={
            "mean": f"{metric}_mean",
            "std":  f"{metric}_std",
            "var":  f"{metric}_var",
            "min":  f"{metric}_min",
            "max":  f"{metric}_max",
            "count":"n_models",
        })
        .reset_index()
    )
    return grouped

def relative_improvement_by_group(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    def _ri(group):
        s = group[metric].dropna()
        if s.empty:
            return pd.Series({f"{metric}_best": None, f"{metric}_worst": None, f"{metric}_RI": None})
        best = s.min()
        worst = s.max()
        ri = (worst - best) / worst if worst != 0 else None
        return pd.Series({f"{metric}_best": best, f"{metric}_worst": worst, f"{metric}_RI": ri})

    out = (
        df.groupby(group_col, group_keys=False)
          .apply(_ri, include_groups=False)
          .reset_index()
    )
    return out


# =========================
# PIPELINE
# =========================
def process_model(model: int):
    df_all = load_reeval_summaries(model)
    df_best = best_per_study(df_all)

    if "split" not in df_best.columns or df_best["split"].isna().all():
        raise ValueError(
            "Spalte 'split' fehlt/leer in Reevaluation-Daten. "
            "Dann muss split in deinen Output-Pfaden vorkommen (Split_X) oder im study-Namen stehen."
        )

    stab_r2   = stability_by_group(df_best, "split", "bester_r2_test")
    stab_rmse = stability_by_group(df_best, "split", "bester_rmse_test")
    ri_rmse   = relative_improvement_by_group(df_best, "split", "bester_rmse_test")

    out_path = OUT_DIR / f"Reeval_Metriken_Modell_{model}.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, sheet_name="Alle_Reeval_Zeilen", index=False)
        df_best.to_excel(writer, sheet_name="Best_je_Study", index=False)
        stab_r2.to_excel(writer, sheet_name="Stabil_R2_proSplit", index=False)
        stab_rmse.to_excel(writer, sheet_name="Stabil_RMSE_proSplit", index=False)
        ri_rmse.to_excel(writer, sheet_name="RI_RMSE_proSplit", index=False)

    print(f"[OK] Modell {model}: gespeichert -> {out_path}")


def main():
    process_model(1)
    process_model(2)


if __name__ == "__main__":
    main()