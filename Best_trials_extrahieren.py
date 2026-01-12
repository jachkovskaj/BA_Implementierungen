import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("Ergebnisse_Teil_3")

# Reihenfolgen für Sortierung
REDUCTION_ORDER = ["CBIR", "RegMix"]

# Fixierungen
MODEL_PLAN_FIX = {1: "Halton", 2: "LHS"}
MODEL_SPLIT_FIX = {1: "SPXY", 2: "KS"}

RE_PLAN   = re.compile(r"_(Halton|LHS|Sobol|Taguchi)_", re.I)
RE_SPLIT  = re.compile(r"_(KS|DUPLEX|SPlit|SPXY)_", re.I)
RE_SEED   = re.compile(r"seed[_-]?(\d+)", re.I)
RE_SUBM1  = re.compile(r"Modell_1[._]([1-3])(?=[^0-9]|$)", re.I)

RE_RED    = re.compile(r"_(CBIR|RegMix)_", re.I)
RE_CONF   = re.compile(r"_CONF_(\d+)", re.I)


def norm_plan(p: str) -> str:
    p = p.lower()
    if p == "lhs": return "LHS"
    if p == "sobol": return "Sobol"
    if p == "taguchi": return "Taguchi"
    if p == "halton": return "Halton"
    return p

def norm_split(s: str) -> str:
    m = {"ks": "KS", "duplex": "DUPLEX", "split": "SPlit", "spxy": "SPXY"}
    return m.get(s.lower(), s)

def norm_red(r: str) -> str:
    r = r.lower()
    if r == "cbir": return "CBIR"
    if r == "regmix": return "RegMix"
    return r


def parse_study_name(study_name: str, model: int):
    m_plan = RE_PLAN.search(study_name)
    plan = norm_plan(m_plan.group(1)) if m_plan else None

    if model == 1:
        m_sub = RE_SUBM1.search(study_name)
        sub = m_sub.group(1) if m_sub else None
        modell_label = f"1.{sub}" if sub else "1"
    else:
        sub = None
        modell_label = "2"

    m_split = RE_SPLIT.search(study_name)
    split = norm_split(m_split.group(1)) if m_split else None

    m_seed = RE_SEED.search(study_name)
    seed = int(m_seed.group(1)) if m_seed else None

    m_red = RE_RED.search(study_name)
    red = norm_red(m_red.group(1)) if m_red else None

    m_conf = RE_CONF.search(study_name)
    conf = int(m_conf.group(1)) if m_conf else None

    return plan, modell_label, sub, split, seed, red, conf


def pick_study_excel(study_dir: Path) -> Path | None:
    preferred = study_dir / f"{study_dir.name}.xlsx"
    if preferred.is_file():
        return preferred
    xlsx = sorted(study_dir.glob("*.xlsx"), key=lambda p: p.stat().st_size, reverse=True)
    return xlsx[0] if xlsx else None


def find_best_trial_in_study(study_file: Path):
    try:
        df = pd.read_excel(study_file, engine="openpyxl")
    except Exception:
        df = pd.read_excel(study_file)

    cols = {str(c).strip().lower(): c for c in df.columns}
    if "state" not in cols or "value" not in cols:
        return None

    c_state = cols["state"]
    c_value = cols["value"]

    dfc = df[df[c_state].astype(str).str.upper() == "COMPLETE"]
    if dfc.empty:
        return None

    idx = dfc[c_value].astype(float).idxmin()
    row = dfc.loc[idx]

    if "number" in cols:
        trial = int(row[cols["number"]])
    elif "trial_id" in cols:
        trial = int(row[cols["trial_id"]])
    else:
        trial = int(idx)

    best_value = float(row[c_value])
    return trial, best_value


def find_metrics_csv(metrics_dir: Path, trial_number: int, *, prefer_reeval: bool) -> Path | None:
    metrics_dir = Path(metrics_dir)

    if prefer_reeval:
        cands = sorted(metrics_dir.glob(f"metrics_{trial_number}__Reeval*.csv"))
        if cands:
            return cands[-1]
        cands = sorted(metrics_dir.glob("metrics_*__Reeval*.csv"),
                       key=lambda p: p.stat().st_size, reverse=True)
        return cands[0] if cands else None

    exact = metrics_dir / f"metrics_{trial_number}.csv"
    if exact.is_file():
        return exact
    cands = [p for p in metrics_dir.glob(f"metrics_{trial_number}*.csv") if "__Reeval" not in p.name]
    if cands:
        cands.sort()
        return cands[-1]
    cands = [p for p in metrics_dir.glob("metrics_*.csv") if "__Reeval" not in p.name]
    if cands:
        cands.sort(key=lambda p: p.stat().st_size, reverse=True)
        return cands[0]
    return None


def read_metrics_any_format(csv_path: Path):
    out = {"rmse_val": np.nan, "r2_val": np.nan, "rmse_test": np.nan, "r2_test": np.nan}
    if csv_path is None or not Path(csv_path).is_file():
        return out

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # long-format: dataset/rmse/r2
    if {"dataset", "rmse", "r2"}.issubset(df.columns):
        norm = {
            "train": "train", "training": "train",
            "val": "validation", "valid": "validation", "validation": "validation",
            "test": "test", "testing": "test",
        }
        ds = df["dataset"].astype(str).str.strip().str.lower().map(norm)
        df2 = df.copy()
        df2["dataset"] = ds

        g = (df2.dropna(subset=["dataset"])
               .groupby("dataset", sort=False)[["rmse", "r2"]]
               .agg("last"))

        if "validation" in g.index:
            out["rmse_val"] = float(g.loc["validation", "rmse"])
            out["r2_val"]   = float(g.loc["validation", "r2"])
        if "test" in g.index:
            out["rmse_test"] = float(g.loc["test", "rmse"])
            out["r2_test"]   = float(g.loc["test", "r2"])
        return out

    # wide-format fallback
    def last(col):
        return float(df[col].iloc[-1]) if col in df.columns and len(df[col]) else np.nan

    out["rmse_val"]  = last("rmse_validation")
    out["r2_val"]    = last("r2_validation")
    out["rmse_test"] = last("rmse_test")
    out["r2_test"]   = last("r2_test")
    return out


def best_trials_reduction(model: int):
    fixed_plan = MODEL_PLAN_FIX[model]
    fixed_split = MODEL_SPLIT_FIX[model]

    rows = []
    n_seen = 0
    n_kept = 0

    for study_dir in sorted(ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        name = study_dir.name
        name_u = name.upper()

        # Modellfilter
        if model == 1 and "MODELL_1" not in name_u:
            continue
        if model == 2 and "MODELL_2" not in name_u:
            continue

        n_seen += 1

        study_file = pick_study_excel(study_dir)
        if study_file is None:
            continue

        plan, modell_label, sub, split, seed, red, conf = parse_study_name(name, model)

        # Fix: Plan + Split
        if plan != fixed_plan:
            continue
        if split != fixed_split:
            continue

        # Reduktionsmethode muss da sein
        if red not in REDUCTION_ORDER:
            continue

        # CBIR: nur CONF-Studies; RegMix: ohne Einschränkung
        if red == "CBIR" and conf is None:
            continue

        best = find_best_trial_in_study(study_file)
        if best is None:
            continue
        trial_number, best_value = best

        metrics_dir = study_dir / "metrics"
        if not metrics_dir.is_dir():
            metrics_dir = study_dir

        metrics_csv_global = find_metrics_csv(metrics_dir, trial_number, prefer_reeval=False)
        if metrics_csv_global is None:
            print(f"[SKIP] Keine GLOBAL metrics gefunden: {study_dir}")
            continue

        metrics_csv_own = find_metrics_csv(metrics_dir, trial_number, prefer_reeval=True)

        m_global = read_metrics_any_format(metrics_csv_global)
        m_own = read_metrics_any_format(metrics_csv_own) if metrics_csv_own else {"rmse_val": np.nan, "r2_val": np.nan, "rmse_test": np.nan, "r2_test": np.nan}

        rows.append({
            "Study": name,
            "Versuchsplan_fix": plan,
            "Split_fix": split,
            "Reduktion": red,
            "CONF": conf,
            "Modell": modell_label,
            "Seed": seed,
            "Bester_Trial": trial_number,
            "Best_value": best_value,

            # Validation + global test
            "r2_val": m_global["r2_val"],
            "rmse_val": m_global["rmse_val"],
            "r2_test_global": m_global["r2_test"],
            "rmse_test_global": m_global["rmse_test"],

            # own test aus __Reeval (falls vorhanden)
            "r2_test_own": (m_own["r2_test"] if metrics_csv_own else np.nan),
            "rmse_test_own": (m_own["rmse_test"] if metrics_csv_own else np.nan),

            "metrics_file_global": str(metrics_csv_global),
            "metrics_file_own": (str(metrics_csv_own) if metrics_csv_own else None),
        })
        n_kept += 1

    out_dir = Path("Ergebnisse_Summary_Datenreduktion")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"BestTrials_Modell_{model}_Datenreduktion.xlsx"

    df_out = pd.DataFrame(rows)

    # Sortierung: Reduktion -> Modell -> Seed -> CONF
    red_rank = {r: i for i, r in enumerate(REDUCTION_ORDER)}

    def modell_key(m):
        parts = str(m).split(".")
        major = int(parts[0]) if parts[0].isdigit() else 999
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (major, minor)

    if not df_out.empty:
        df_out["__red_rank"] = df_out["Reduktion"].map(red_rank).fillna(999).astype(int)
        df_out["__modell_key"] = df_out["Modell"].apply(modell_key)
        df_out["__seed"] = df_out["Seed"].fillna(999999).astype(int)
        df_out["__conf"] = df_out["CONF"].fillna(999999).astype(int)

        df_out = df_out.sort_values(
            by=["__red_rank", "__modell_key", "__seed", "__conf"],
            ascending=True
        ).drop(columns=["__red_rank", "__modell_key", "__seed", "__conf"])

    df_out.to_excel(out_file, index=False)
    print(f"[OK] Modell {model}: gesehen={n_seen}, behalten={n_kept}, geschrieben={out_file}")


if __name__ == "__main__":
    best_trials_reduction(1)
    best_trials_reduction(2)