import pandas as pd
from pathlib import Path


SUMMARY_DIR = Path("Ergebnisse_Summary_Reeval")


def load_besttrials(model: int) -> pd.DataFrame:
    """
    Lädt die BestTrials-Tabelle für ein Modell.
    """
    path = SUMMARY_DIR / f"BestTrials_Modell_{model}.xlsx"
    df = pd.read_excel(path)
    return df


def add_generalisation_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt pro Modell zwei Generalisierungs-Gaps hinzu:
    - gap_R2_val_test: |R2_val - R2_test|
    - gap_RMSE_val_test: |RMSE_test - RMSE_val|
    """
    df = df.copy()
    df["gap_R2_val_test"] = (df["bester_r2"] - df["bester_r2_test"]).abs()
    df["gap_RMSE_val_test"] = (df["bester_rmse_test"] - df["bester_rmse_val"]).abs()
    return df


def stability_by_plan(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Berechnet die Stabilität eines Versuchsplans bezüglich einer Kennzahl.

    metric: Spaltenname, z.B. 'bester_r2_test' oder 'bester_rmse_test'

    Gibt pro Versuchsplan:
        - Mittelwert
        - Standardabweichung
        - Varianz
        - Minimum
        - Maximum
        - Anzahl Modelle
    zurück.
    """
    grouped = (
        df.groupby("Versuchsplan")[metric]
        .agg(["mean", "std", "var", "min", "max", "count"])
        .rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "var": f"{metric}_var",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
                "count": "n_models",
            }
        )
        .reset_index()
    )
    return grouped


def relative_improvement_by_plan(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Relative Improvement (RI) eines Plans bzgl. eines Fehlers (typisch RMSE):
        RI = (worst - best) / worst
    """
    def _ri(group):
        series = group[metric].dropna()
        if series.empty:
            return pd.Series(
                {
                    f"{metric}_best": None,
                    f"{metric}_worst": None,
                    f"{metric}_RI": None,
                }
            )

        best = series.min()    # bei RMSE ist klein gut
        worst = series.max()
        ri = (worst - best) / worst if worst != 0 else None

        return pd.Series(
            {
                f"{metric}_best": best,
                f"{metric}_worst": worst,
                f"{metric}_RI": ri,
            }
        )

    out = (
        df.groupby("Versuchsplan", group_keys=False)
          .apply(_ri, include_groups=False)
          .reset_index()
    )

    return out

def compute_cpti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-Plan Transferability Index (CPTI) pro Modell und Plan.

    Definition hier auf R²-Basis:

        CPTI_R2 = R2_test_global / R2_test_eigene_Daten

    Zusätzlich als RMSE-Analogie:

        CPTI_RMSE = RMSE_test_eigene_Daten / RMSE_test_global

    (Da bei RMSE kleinere Werte besser sind, bedeutet CPTI_RMSE ~ 1
    gute Übertragbarkeit, >1 = Modell ist auf globalem Testset besser.)
    """
    df = df.copy()

    eps = 1e-8
    # R²-CPTI – nur da berechnen, wo beide Werte existieren
    df["CPTI_R2"] = df["bester_r2_test"] / (df["bester_r2_test_eigene_Daten"] + eps)

    # RMSE-CPTI
    df["CPTI_RMSE"] = df["bester_rmse_test_eigene_Daten"] / (df["bester_rmse_test"] + eps)

    # Planweise Statistik
    cpti_plan = (
        df.groupby("Versuchsplan")[["CPTI_R2", "CPTI_RMSE"]]
        .agg(["mean", "std", "min", "max", "count"])
    )
    # MultiIndex-Spalten etwas hübscher machen
    cpti_plan.columns = ["_".join(col).strip() for col in cpti_plan.columns.values]
    cpti_plan = cpti_plan.reset_index()

    return df, cpti_plan


def process_model(model: int):
    """
    Lädt BestTrials für ein Modell, berechnet alle Metriken und
    speichert:
      - Zeilenweise Metriken (pro Modell)
      - Planweise zusammengefasste Metriken
    """
    df = load_besttrials(model)

    # 1) Generalisierungs-Gaps (Val vs. globaler Test)
    df = add_generalisation_gap(df)

    # 2) Stabilität pro Plan für R²_test und RMSE_test
    stab_r2_global = stability_by_plan(df, "bester_r2_test")
    stab_rmse_global = stability_by_plan(df, "bester_rmse_test")

    # 3) Stabilität pro Plan für Reeval-Test (eigene Daten, falls vorhanden)
    stab_r2_own = stability_by_plan(df, "bester_r2_test_eigene_Daten")
    stab_rmse_own = stability_by_plan(df, "bester_rmse_test_eigene_Daten")

    # 4) Relative Improvement pro Plan (globaler Test-RMSE)
    ri_global = relative_improvement_by_plan(df, "bester_rmse_test")

    # Optional: RI für Reeval-RMSE
    ri_own = relative_improvement_by_plan(df, "bester_rmse_test_eigene_Daten")

    # 5) CPTI (Generalisation von „eigene“ → „globale“ Testdaten)
    df_with_cpti, cpti_plan = compute_cpti(df)

    # Alles in eine Excel-Datei schreiben
    out_path = SUMMARY_DIR / f"Modellbewertung_{model}_mit_Metriken.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_with_cpti.to_excel(writer, sheet_name="Modelle_mit_Metriken", index=False)
        stab_r2_global.to_excel(writer, sheet_name="Stabil_R2_global", index=False)
        stab_rmse_global.to_excel(writer, sheet_name="Stabil_RMSE_global", index=False)
        stab_r2_own.to_excel(writer, sheet_name="Stabil_R2_eigene", index=False)
        stab_rmse_own.to_excel(writer, sheet_name="Stabil_RMSE_eigene", index=False)
        ri_global.to_excel(writer, sheet_name="RI_RMSE_global", index=False)
        ri_own.to_excel(writer, sheet_name="RI_RMSE_eigene", index=False)
        cpti_plan.to_excel(writer, sheet_name="CPTI_je_Plan", index=False)

    print(f"[OK] Metriken für Modell {model} gespeichert unter: {out_path}")


if __name__ == "__main__":
    # Für Modell 1 und 2 durchlaufen
    process_model(1)
    process_model(2)