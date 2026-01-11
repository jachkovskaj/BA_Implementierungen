import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# Konfiguration
# ============================================================
BASE_DIR_T2 = Path("Ergebnisse_Teil_2")
BASE_DIR_T3 = Path("Ergebnisse_Teil_3")
OUTPUT_DIR  = Path("Ergebnisplots_Vergleich_RegMix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Farben wie gewünscht:
C_VAL  = "#00354E"   # dunkelblau
C_TEST = "#717E86"   # grau

# Hintergrundfarben nach Modell
BG_MODEL = {
    1: "#eef7ff",  # blau
    2: "#fff3e0",  # gelb
}

# Fixierungen
PLAN_FIX  = {1: "Halton", 2: "LHS"}
SPLIT_FIX = {1: "SPXY",   2: "KS"}

# ============================================================
# Regex / Parser
# ============================================================
RE_MODEL = re.compile(r"Modell_([\d.]+)")
RE_SEED  = re.compile(r"seed_(\d+)", re.I)
RE_SPLIT = re.compile(r"_(KS|DUPLEX|SPlit|SPXY)_", re.I)

def norm_split(token: str) -> str:
    m = {"ks": "KS", "duplex": "DUPLEX", "split": "SPlit", "spxy": "SPXY"}
    return m.get(token.lower(), token)

def parse_model(s: str) -> str | None:
    m = RE_MODEL.search(s)
    return m.group(1) if m else None

def parse_major(model_str: str | None) -> int | None:
    if not model_str:
        return None
    try:
        return int(model_str.split(".")[0])
    except Exception:
        return None

def parse_seed(s: str) -> int | None:
    m = RE_SEED.search(s)
    return int(m.group(1)) if m else None

def parse_split(s: str) -> str | None:
    m = RE_SPLIT.search(s)
    return norm_split(m.group(1)) if m else None

# ============================================================
# IO helpers
# ============================================================
def pick_study_excel(study_dir: Path) -> Path | None:
    preferred = study_dir / f"{study_dir.name}.xlsx"
    if preferred.is_file():
        return preferred
    xlsx = sorted(study_dir.glob("*.xlsx"), key=lambda p: p.stat().st_size, reverse=True)
    return xlsx[0] if xlsx else None

def best_trial_from_study_xlsx(study_xlsx: Path) -> int | None:
    try:
        df = pd.read_excel(study_xlsx, engine="openpyxl")
    except Exception:
        return None
    if not {"state", "value"}.issubset(df.columns):
        return None
    dfc = df[df["state"] == "COMPLETE"]
    if dfc.empty:
        return None
    row = dfc.loc[dfc["value"].idxmin()]
    if "number" in dfc.columns:
        return int(row["number"])
    if "trial_id" in dfc.columns:
        return int(row["trial_id"])
    return None

def find_metrics_csv(metrics_dir: Path, trial: int) -> Path | None:
    direct = metrics_dir / f"metrics_{trial}.csv"
    if direct.is_file():
        return direct
    cands = sorted([p for p in metrics_dir.glob("metrics_*.csv")
                    if str(trial) in p.name and "__Reeval" not in p.name])
    if cands:
        return cands[-1]
    cands = sorted([p for p in metrics_dir.glob("metrics_*.csv") if "__Reeval" not in p.name])
    return cands[-1] if cands else None

def read_metrics(metrics_csv: Path) -> dict:
    m = pd.read_csv(metrics_csv)
    m.columns = [str(c).strip().lower() for c in m.columns]
    out = {
        "train": {"rmse": np.nan, "r2": np.nan},
        "validation": {"rmse": np.nan, "r2": np.nan},
        "test": {"rmse": np.nan, "r2": np.nan},
    }

    if {"dataset", "rmse", "r2"}.issubset(m.columns):
        norm = {
            "train": "train", "training": "train",
            "valid": "validation", "validation": "validation", "val": "validation",
            "test": "test", "testing": "test",
        }
        m["dataset"] = m["dataset"].astype(str).str.strip().str.lower().map(norm)
        g = (m.dropna(subset=["dataset"])
               .groupby("dataset", sort=False)[["rmse", "r2"]]
               .agg("last"))
        for ds in out:
            if ds in g.index:
                out[ds]["rmse"] = float(g.loc[ds, "rmse"])
                out[ds]["r2"]   = float(g.loc[ds, "r2"])
        return out

    def get(col):
        return float(m[col].iloc[-1]) if col in m.columns and len(m[col]) else np.nan

    for ds in out:
        out[ds]["rmse"] = get(f"rmse_{ds}")
        out[ds]["r2"]   = get(f"r2_{ds}")
    return out

# ============================================================
# Teil-2 Index (Baseline) bauen: (model_str, split, seed) -> dir
# ============================================================
def build_t2_index() -> dict:
    idx = {}
    for d in BASE_DIR_T2.iterdir():
        if not d.is_dir():
            continue
        name = d.name

        # Baseline: keine Datenreduktion / kein RegMix
        if "_CONF_" in name.upper():
            continue
        if "_CBIR_" in name.upper() or "_REGMIX_" in name.upper():
            continue

        model = parse_model(name)
        seed  = parse_seed(name)
        split = parse_split(name)
        if not model or seed is None or not split:
            continue

        major = parse_major(model)
        if major not in (1, 2):
            continue

        if split != SPLIT_FIX[major]:
            continue

        key = (model, split, seed)

        plan_req = PLAN_FIX[major]
        has_plan = f"_{plan_req}_" in name
        if key not in idx:
            idx[key] = d
        else:
            prev_has_plan = f"_{plan_req}_" in idx[key].name
            if has_plan and not prev_has_plan:
                idx[key] = d
    return idx

# ============================================================
# Matching Teil3(RegMix) -> Teil2(Baseline)
# ============================================================
def collect_pairs_for_model(major: int, t2_index: dict) -> pd.DataFrame:
    records = []

    for d in BASE_DIR_T3.iterdir():
        if not d.is_dir():
            continue
        name = d.name

        # NUR RegMix-Studies (kein CONF nötig)
        if "_REGMIX_" not in name.upper():
            continue

        model = parse_model(name)
        seed  = parse_seed(name)
        split = parse_split(name)

        if not model or seed is None or not split:
            continue
        if parse_major(model) != major:
            continue
        if split != SPLIT_FIX[major]:
            continue

        key = (model, split, seed)
        t2_dir = t2_index.get(key)
        if t2_dir is None:
            continue

        # Teil 3 best trial + metrics
        t3_xlsx = pick_study_excel(d)
        if t3_xlsx is None:
            continue
        t3_trial = best_trial_from_study_xlsx(t3_xlsx)
        if t3_trial is None:
            continue
        t3_mdir = d / "metrics"
        if not t3_mdir.is_dir():
            continue
        t3_csv = find_metrics_csv(t3_mdir, t3_trial)
        if t3_csv is None:
            continue
        m3 = read_metrics(t3_csv)

        # Teil 2 best trial + metrics
        t2_xlsx = pick_study_excel(t2_dir)
        if t2_xlsx is None:
            continue
        t2_trial = best_trial_from_study_xlsx(t2_xlsx)
        if t2_trial is None:
            continue
        t2_mdir = t2_dir / "metrics"
        if not t2_mdir.is_dir():
            continue
        t2_csv = find_metrics_csv(t2_mdir, t2_trial)
        if t2_csv is None:
            continue
        m2 = read_metrics(t2_csv)

        label = f"Seed\n{seed}"

        records.append({
            "label": label,
            "major": major,
            "model": model,
            "seed": seed,

            "t2_rmse_val": m2["validation"]["rmse"],
            "t3_rmse_val": m3["validation"]["rmse"],
            "t2_rmse_test": m2["test"]["rmse"],
            "t3_rmse_test": m3["test"]["rmse"],

            "t2_r2_val": m2["validation"]["r2"],
            "t3_r2_val": m3["validation"]["r2"],
            "t2_r2_test": m2["test"]["r2"],
            "t3_r2_test": m3["test"]["r2"],
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(by=["major", "model", "seed"]).reset_index(drop=True)
    return df

# ============================================================
# Plot helpers
# ============================================================
def _tight_ylim(ax, arrays, metric: str, r2_top: float | None = None):
    vals = np.concatenate([np.asarray(a, float) for a in arrays])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return

    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin

    if metric == "rmse":
        margin = 0.03 * span if span > 0 else max(1e-6, 0.03 * abs(vmax))
        ax.set_ylim(vmin - margin, vmax + margin)
    else:
        margin = 0.015
        low = max(-1.0, vmin - margin)
        high = min(r2_top if r2_top is not None else 1.0, vmax + margin)
        ax.set_ylim(low, high)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=40))
    ax.grid(which="major", linestyle="--", alpha=0.6)
    ax.grid(which="minor", linestyle=":", alpha=0.25)

# ============================================================
# Plot: gemeinsamer Scatter für beide Modelle
# ============================================================
def plot_metric_both_models(df_all: pd.DataFrame, metric: str):
    if df_all.empty:
        print("[Hinweis] Keine gematchten Daten für Modell 1/2.")
        return

    df = df_all.sort_values(by=["major", "model", "seed"]).reset_index(drop=True)

    labels = df["label"].tolist()
    majors = df["major"].to_numpy(int)
    x = np.arange(len(labels))

    if metric == "rmse":
        y_t2_val  = df["t2_rmse_val"].to_numpy(float)
        y_t3_val  = df["t3_rmse_val"].to_numpy(float)
        y_t2_test = df["t2_rmse_test"].to_numpy(float)
        y_t3_test = df["t3_rmse_test"].to_numpy(float)

        title = "RMSE – Modell 1 & Modell 2 – Mit und ohne Datenerweiterung (RegMix)"
        fname = OUTPUT_DIR / "RMSE_Modell_1_und_2_Mit_und_ohne_RegMix.png"
        ylabel = "RMSE"
        r2_top = None

    elif metric == "r2":
        y_t2_val  = df["t2_r2_val"].to_numpy(float)
        y_t3_val  = df["t3_r2_val"].to_numpy(float)
        y_t2_test = df["t2_r2_test"].to_numpy(float)
        y_t3_test = df["t3_r2_test"].to_numpy(float)

        title = "R² – Modell 1 & Modell 2 – Mit und ohne Datenerweiterung (RegMix)"
        fname = OUTPUT_DIR / "R2_Modell_1_und_2_Mit_und_ohne_RegMix.png"
        ylabel = "R²"
        r2_top = 1.2
    else:
        raise ValueError("metric muss 'rmse' oder 'r2' sein.")

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, maj in enumerate(majors):
        ax.axvspan(i - 0.5, i + 0.5, color=BG_MODEL.get(int(maj), "#ffffff"), alpha=0.85, zorder=0)

    # Teil2: gefüllt, Teil3: hohl
    ax.scatter(x, y_t2_val,  color=C_VAL,  marker="o", s=40, label="Ohne RegMix – Validation", zorder=3)
    ax.scatter(x, y_t3_val,  facecolors="none", edgecolors=C_VAL, marker="o", s=55, linewidths=1.8,
               label="Mit RegMix – Validation", zorder=4)

    ax.scatter(x, y_t2_test, color=C_TEST, marker="s", s=40, alpha=0.85, label="Ohne RegMix – Test", zorder=3)
    ax.scatter(x, y_t3_test, facecolors="none", edgecolors=C_TEST, marker="s", s=55, linewidths=1.8,
               label="Mit RegMix – Test", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _tight_ylim(ax, [y_t2_val, y_t3_val, y_t2_test, y_t3_test], metric=metric, r2_top=r2_top)

    if metric == "r2":
        ax.set_ylim(top=1.3)
        ax.legend(loc="upper right")
    else:
        ax.legend()

    plt.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {fname}")

# ============================================================
# Main
# ============================================================
def main():
    t2_index = build_t2_index()

    df1 = collect_pairs_for_model(1, t2_index)
    df2 = collect_pairs_for_model(2, t2_index)

    df_all = pd.concat([df1, df2], ignore_index=True)
    if df_all.empty:
        print("[Hinweis] Keine gematchten Daten gefunden.")
        return

    df_all.to_csv(OUTPUT_DIR / "matched_modell_1_und_2_regmix.csv", index=False)

    plot_metric_both_models(df_all, metric="rmse")
    plot_metric_both_models(df_all, metric="r2")

if __name__ == "__main__":
    main()