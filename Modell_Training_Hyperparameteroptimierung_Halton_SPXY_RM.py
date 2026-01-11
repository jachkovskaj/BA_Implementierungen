import sys
try:
    import setuptools._distutils as _distutils
    sys.modules["distutils"] = _distutils
except Exception as e:
    print("Warnung: distutils-Shim konnte nicht geladen werden:", e)

import atexit
import glob
import math
import os
import signal
import keras
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import logging
from tensorflow.keras import backend as K
from optuna_integration import TFKerasPruningCallback
from packaging import version
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors


optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.enable_default_handler()
logger = logging.getLogger("optuna")
logger.setLevel(logging.INFO)

class Optimization:
    """
    Manages an optimization process for a machine learning model.

    This class is designed to manage and execute the optimization process
    for a machine learning model. It provides attributes to initialize
    specific parameters such as study name, random seed, patience, number
    of epochs, number of optimization trials, and dataset. It facilitates
    an organized approach to model tuning and experimentation.

    :ivar epochs: Number of epochs to train the model.
    :type epochs: int
    :ivar n_trials: Number of optimization trials to perform.
    :type n_trials: int
    :ivar patience: Number of consecutive epochs without improvement
        allowed before stopping training.
    :type patience: int
    :ivar seed: Random seed value to ensure reproducibility.
    :type seed: int
    :ivar study_name: Name of the study for tracking optimization results.
    :type study_name: str
    :ivar -data: Dataset to be used for model training and validation.
    :type -data: Any
    :ivar model_file: Path to save or load the trained model.
    :type model_file: str
    """
    def __init__(self, study_name, seed, patience, epochs, n_trials, train_data, test_data):
        self.epochs = epochs
        self.n_trials = n_trials
        self.patience = patience
        self.seed = seed
        self.study_name = study_name
        self.train_data = train_data
        self.test_data = test_data
        self.model_file = ""


def initialize(study_name, seed, patience, epochs, n_trials, train_data, test_data):
    """
    Initializes the global optimization process for hyperparameter tuning using the
    provided input parameters. Validates TensorFlow version compatibility and ensures
    the unique naming of the optimization study. Handles the creation of the global
    hybrid optimization object and performs initializations necessary for the process.

    :param study_name: Name of the hyperparameter optimization study file.
    :param seed: Seed value for ensuring reproducibility of results.
    :param patience: Number of epochs with no improvement after which training is stopped.
    :param epochs: Total number of training epochs.
    :param n_trials: Number of hyperparameter optimization trials to perform.
    :param data: Dataset used for training and evaluation during optimization.
    :return: None
    """
    global hpoptimize
    hpoptimize = Optimization(study_name, seed, patience, epochs, n_trials, train_data, test_data)

    # Tensorflow Version Überprüfen
    if version.parse(tf.__version__) < version.parse("2.11.0"):
        raise RuntimeError("tensorflow>=2.11.0 is required for this example.")
    # Ask user for Study name
    if os.path.isfile(hpoptimize.study_name + ".sqlite3") is True:
        print("This file already exists! Choose a different name for the Study")
        hpoptimize.study_name = input("Study name: ")
    hpoptimize.model_file = 'checkpoint.model.keras'


# Global variable to store the current model
current_model = None


def save_model_on_interrupt(signum, frame):
    """
    Handles the interrupt signal to save the current model if it exists.

    This function is designed to be invoked when an interrupt signal is
    received, such as a SIGINT (e.g., triggered by pressing Ctrl+C). It
    checks if a global variable `current_model` is defined, and if so,
    it saves the model to a file named 'interrupted_model.keras'.
    If no model is available, it notifies the user that there is no model
    to save.

    :param signum: Signal number indicating the received signal.
    :type signum: int
    :param frame: Current stack frame at the time of interruption.
    :type frame: types.FrameType
    """
    global current_model
    if current_model:
        print("\nProgram interrupted. Saving the current model...")
        current_model.save("interrupted_model.keras")
        print("Model saved as 'interrupted_model.keras'.")
    else:
        print("\nNo model to save.")


# Register the signal handler
signal.signal(signal.SIGINT, save_model_on_interrupt)

def parse_mplstyle(style_file_path):
    """
    Parses a Matplotlib style file (.mplstyle) and converts it into a dictionary
    of style settings. Each key-value pair represents a specific style setting
    defined in the file. Empty lines and comments in the file are ignored during
    parsing.

    :param style_file_path: Path to the Matplotlib style file (.mplstyle) that
        needs to be parsed.

    :return: Dictionary containing the style settings parsed from the file. Each
        key corresponds to a setting, and its value represents the associated
        property.
    :rtype: dict
    """
    with open(style_file_path, 'r') as file:
        style_settings = {}
        for line in file:
            # Ignoriere Kommentare und leere Zeilen
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.strip().split(':', 1)
                style_settings[key.strip()] = value.strip()
    return style_settings


# Plotte History und prediction Value vs. true Value
def plot_graphs(history, trial_number, y_train, train_predict, y_val, val_predict, y_test, test_predict):
    """
    Plots and saves various graphs to evaluate model performance, including RMSE history,
    Huber loss history, and model predictions versus actual values.

    Adjustments are necessary for different data sets:
    - The limits for the x- and y-axes
    - The axis labels

    :param history: Training history object containing the metric and loss values for
                    both training and validation datasets.
    :type history: tf.keras.callbacks.History
    :param trial_number: Identifier for the specific trial to label saved plots.
    :type trial_number: int
    :param y_train: True values for the training dataset for comparison with predictions.
    :type y_train: numpy.ndarray
    :param train_predict: Predicted values corresponding to the training dataset.
    :type train_predict: numpy.ndarray
    :param y_val: True values for the validation dataset for comparison with predictions.
    :type y_val: numpy.ndarray
    :param val_predict: Predicted values corresponding to the validation dataset.
    :type val_predict: numpy.ndarray
    :param y_test: True values for the test dataset for comparison with predictions.
    :type y_test: numpy.ndarray
    :param test_predict: Predicted values corresponding to the test dataset.
    :type test_predict: numpy.ndarray
    :return: None
    """

    #IKV Stil für Grafiken
    plt.rcdefaults
    plt.style.use(["IKV.mplstyle"])

    # Plot History RMSE
    fig, ax = plt.subplots()
    plt.plot(history.history["val_root_mean_squared_error"], label="Validationsdaten")
    plt.plot(history.history["root_mean_squared_error"], label="Trainingsdaten")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_ylabel("RMSE [-]")
    ax.set_xlabel("Epoche  [-]")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(hpoptimize.paths["plots"], f"{trial_number}_RMSE.png"))
    plt.close()

    # Plot History Huber
    fig, ax = plt.subplots()
    plt.plot(history.history["val_loss"], label="Validationsdaten")
    plt.plot(history.history["loss"], label="Trainingsdaten")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_ylabel("Huber [-]")
    ax.set_xlabel("Epoche  [-]")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(hpoptimize.paths["plots"], f"{trial_number}_Huber_Loss.png"))
    plt.close()

    # Plot Predict vs True
    plt.scatter(y_train, train_predict, label="Trainingsdaten")
    plt.scatter(y_test, test_predict, label="Testdaten")
    plt.scatter(y_val, val_predict, label="Validationsdaten")
    ax = plt.gca()
    ax.legend(loc='upper left')
    plt.xlabel('Reale Fließfronten [-]', fontsize=15) # Adjust Label
    ax.set_xlim([0, 105]) # Adjust limit for the x-Axis
    ax.set_xticks(np.arange(0, 105, step=10)) # Adjust limit for the steps on the x-Axis
    plt.ylabel('Vorhergesagte Fließfronten [-]', fontsize=15) # Adjust label
    ax.set_ylim([0, 105])  # Adjust limit for the y-Axis
    ax.set_yticks(np.arange(0, 105, step=10)) # Adjust limit for the steps on the y-Axis
    line = mlines.Line2D([0, 1], [0, 1], color="black", alpha=0.8)
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.tight_layout()
    plt.savefig(os.path.join(hpoptimize.paths["plots"], f"{trial_number}_Predict_vs._True.png"))
    plt.close()


def apply_custom_chart_style(chart_all_data, y_train, y_val, y_test, train_predict, val_predict, test_predict, combined_df, worksheet_all_data):
    """
    Applies a custom chart style to the given chart object. This function formats the chart by
    setting titles, axes, plot area, chart area, and gridlines appearance based on calculated
    maximum values derived from true and predicted data.

    :param chart_all_data: The chart object to which the styling is applied.
    :param y_train: True values for the training dataset.
    :param y_val: True values for the validation dataset.
    :param y_test: True values for the testing dataset.
    :param train_predict: Predicted values for the training dataset.
    :param val_predict: Predicted values for the validation dataset.
    :param test_predict: Predicted values for the testing dataset.
    :param combined_df: Combined DataFrame used for deriving insights for styling.
    :param worksheet_all_data: Worksheet object used for displaying the chart.
    :return: None
    """
    max_value = max(
        y_train.max(), y_val.max(), y_test.max(),
        train_predict.max(), val_predict.max(), test_predict.max())
    max_value_rounded = math.ceil(max_value)

    chart_all_data.set_title({
        "name": "True vs Predicted Values",
        "name_font": {"size": 14, "bold": True, "color": "black"}
    })

    chart_all_data.set_x_axis({
        "name": "True Values [mm]",
        "name_font": {"size": 12, "bold": False, "color": "black"},
        "num_font": {"size": 10, "color": "black"},
        "line": {"color": "black", "width": 1.0},
        "major_gridlines": {"visible": True, "line": {"color": "black", "width": 0.5}},
        "min": 0,
        "max": max_value_rounded
    })

    chart_all_data.set_y_axis({
        "name": "Predicted Values [mm]",
        "name_font": {"size": 12, "bold": False, "color": "black"},
        "num_font": {"size": 10, "color": "black"},
        "line": {"color": "black", "width": 1.0},
        "major_gridlines": {"visible": True, "line": {"color": "black", "width": 0.5}},
        "min": 0,
        "max": max_value_rounded
    })

    chart_all_data.set_plotarea({
        "border": {"color": "black", "width": 0.5},
        "fill": {"color": "white"}
    })
    chart_all_data.set_chartarea({
        "border": {"color": "black", "width": 0.5},
        "fill": {"color": "white"}
    })


def save_trial_results_with_dynamic_style(trial_number, y_train, train_predict, y_test, test_predict, y_val, val_predict, history):
    """
    Speichert die Trial-Ergebnisse_Teil_2 in eine Excel-Datei und erzeugt einen Scatter-Chart
    (Train/Val/Test) mit 45°-Linie + vertikaler/horizontaler Hilfslinie.
    Die Dummy-Punkte für die Linien werden – wie im Reeval-Skript – in die Spalten
    'Chart_X' und 'Chart_Y' geschrieben, NICHT als zusätzliche Zeilen am Tabellenende.
    """

    output_file = os.path.join(hpoptimize.paths["metrics"], f"Trial_{trial_number}.xlsx")

    # DataFrames für Train/Val/Test
    train_df = pd.DataFrame({
        "True Values Train": y_train.to_numpy().flatten(),
        "Predictions Train": train_predict.flatten()
    })
    valid_df = pd.DataFrame({
        "True Values Validation": y_val.to_numpy().flatten(),
        "Predictions Validation": val_predict.flatten()
    })
    test_df = pd.DataFrame({
        "True Values Test": y_test.to_numpy().flatten(),
        "Predictions Test": test_predict.flatten()
    })

    # Kombiniert nach Spalten (wie bisher)
    combined_df = pd.concat([train_df, valid_df, test_df], axis=1)

    # --- Chart-Hilfsspalten wie im Reeval-Skript ---
    # Chart_X / Chart_Y als zusätzliche Spalten (keine neuen Zeilen)
    combined_df["Chart_X"] = np.nan
    combined_df["Chart_Y"] = np.nan

    # max_value aus allen True/Predicted-Spalten
    max_value = math.ceil(np.nanmax([
        combined_df["True Values Train"].max(),
        combined_df["True Values Validation"].max(),
        combined_df["True Values Test"].max(),
        combined_df["Predictions Train"].max(),
        combined_df["Predictions Validation"].max(),
        combined_df["Predictions Test"].max(),
    ]))

    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0

    # Dummy-Punkte (identisch zur Reeval-Logik)
    dummy_points = [
        (0.0,       0.0),        # für 45°-Linie Start
        (max_value, max_value),  # für 45°-Linie Ende
        (max_value, 0.0),        # vertikale Linie x = max_value (Start)
        (max_value, max_value),  # vertikale Linie x = max_value (Ende)
        (0.0,       max_value),  # horizontale Linie y = max_value (Start)
        (max_value, max_value),  # horizontale Linie y = max_value (Ende)
    ]

    n_dummy = min(len(dummy_points), len(combined_df))
    for i in range(n_dummy):
        combined_df.at[i, "Chart_X"] = dummy_points[i][0]
        combined_df.at[i, "Chart_Y"] = dummy_points[i][1]

    # History-Sheet vorbereiten
    history_df = pd.DataFrame({
        "Epoch": range(1, len(history.history["loss"]) + 1),
        "Train RMSE": history.history["root_mean_squared_error"],
        "Validation RMSE": history.history["val_root_mean_squared_error"],
        "Train Loss": history.history["loss"],
        "Validation Loss": history.history["val_loss"]
    })

    # --- Schreiben & Chart erzeugen ---
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        # All_Data inkl. Chart_X / Chart_Y
        combined_df.to_excel(writer, sheet_name="All_Data", index=False, header=True)
        # Metrics / History
        history_df.to_excel(writer, sheet_name="Metrics", index=False)

        workbook = writer.book
        worksheet_all_data = writer.sheets["All_Data"]
        chart_all_data = workbook.add_chart({"type": "scatter"})

        # Spaltenindizes ermitteln
        col_true_train = combined_df.columns.get_loc("True Values Train")
        col_pred_train = combined_df.columns.get_loc("Predictions Train")
        col_true_val   = combined_df.columns.get_loc("True Values Validation")
        col_pred_val   = combined_df.columns.get_loc("Predictions Validation")
        col_true_test  = combined_df.columns.get_loc("True Values Test")
        col_pred_test  = combined_df.columns.get_loc("Predictions Test")
        col_chart_x    = combined_df.columns.get_loc("Chart_X")
        col_chart_y    = combined_df.columns.get_loc("Chart_Y")

        # Anzahl Zeilen pro Datensatz (für die Test-Punkte kannst du auch len(test_df) benutzen)
        n_rows      = len(combined_df)
        n_train     = len(train_df)
        n_val       = len(valid_df)
        n_test      = len(test_df)
        start_row   = 1  # Excel: 0=Header, 1=erste Datenzeile

        # Train / Val / Test
        for dataset, color, (col_x, col_y, n_used) in [
            ("Train",      "#95BB20", (col_true_train, col_pred_train, n_train)),
            ("Validation", "#717E86", (col_true_val,   col_pred_val,   n_val)),
            ("Test",       "#00354E", (col_true_test,  col_pred_test,  n_test)),
        ]:
            last_row = start_row + n_used - 1
            chart_all_data.add_series({
                "name":       f"{dataset} Data",
                "categories": ["All_Data", start_row, col_x, last_row, col_x],
                "values":     ["All_Data", start_row, col_y, last_row, col_y],
                "marker": {
                    "type":  "circle",
                    "size":  6,
                    "fill":   {"color": color},
                    "border": {"none": True},
                },
            })

        # 45°-Linie (Entries 0 und 1 in Chart_X/Y)
        if n_dummy >= 2:
            chart_all_data.add_series({
                "name": "45° Linie",
                "categories": ["All_Data", 1, col_chart_x, 2, col_chart_x],
                "values":     ["All_Data", 1, col_chart_y, 2, col_chart_y],
                "line": {"color": "black", "dash_type": "dash", "width": 1.0},
                "marker": {"type": "none"},
            })

        # Vertikale Linie x = max_value (Entries 2 und 3)
        if n_dummy >= 4:
            chart_all_data.add_series({
                "name": None,
                "categories": ["All_Data", 3, col_chart_x, 4, col_chart_x],
                "values":     ["All_Data", 3, col_chart_y, 4, col_chart_y],
                "line": {"color": "black", "width": 1.0},
                "marker": {"type": "none"},
            })

        # Horizontale Linie y = max_value (Entries 4 und 5)
        if n_dummy >= 6:
            chart_all_data.add_series({
                "name": None,
                "categories": ["All_Data", 5, col_chart_x, 6, col_chart_x],
                "values":     ["All_Data", 5, col_chart_y, 6, col_chart_y],
                "line": {"color": "black", "width": 1.0},
                "marker": {"type": "none"},
            })

        # Achsen-/Chart-Style
        apply_custom_chart_style(
            chart_all_data,
            y_train, y_val, y_test,
            train_predict, val_predict, test_predict,
            combined_df,
            worksheet_all_data
        )

        # Chart einfügen
        worksheet_all_data.insert_chart("J2", chart_all_data)


def spxy_train_val_split(X, y, test_size=0.111111, metric="euclidean", alpha=0.5):
    """
    SPXY Split nach Galvão et al. (2005): joint X–Y distances.  [oai_citation:1‡ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S003991400500192X?utm_source=chatgpt.com)
    - X: (n_samples, n_features) DataFrame/ndarray
    - y: (n_samples,) oder (n_samples, n_targets)
    - test_size: Anteil für Validation (wie bei dir ~0.111111)
    - metric: "euclidean" (Standard in vielen Umsetzungen)
    - alpha: Gewichtung zwischen X- und y-Distanz (0..1). alpha=0.5 => gleichgewichtet
    Rückgabe: X_train, X_val, y_train, y_val (Indices deterministisch)
    """

    # --- in numpy ---
    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=float)
    if yn.ndim == 1:
        yn = yn.reshape(-1, 1)

    n = Xn.shape[0]
    n_val = int(np.ceil(n * test_size))
    n_train = n - n_val
    if n_train <= 1 or n_val <= 0:
        raise ValueError("test_size führt zu ungültiger Train/Val-Aufteilung.")

    # --- pairwise distances (ohne scipy, robust) ---
    # Distanzmatrix für X
    # D_X[i,j] = ||X_i - X_j||
    diffX = Xn[:, None, :] - Xn[None, :, :]
    DX = np.sqrt(np.sum(diffX * diffX, axis=2))

    # Distanzmatrix für y
    diffY = yn[:, None, :] - yn[None, :, :]
    DY = np.sqrt(np.sum(diffY * diffY, axis=2))

    # --- Normierung wie üblich, damit X und y vergleichbar sind ---
    DX = DX / (DX.max() + 1e-12)
    DY = DY / (DY.max() + 1e-12)

    # --- Joint distance ---
    # Viele Beschreibungen verwenden DX + DY (gleichgewichtet).
    # Hier: gewichtete Summe, falls du testen willst:
    D = alpha * DX + (1.0 - alpha) * DY

    # --- KS-artige Selektion: max-min distance Auswahl für TRAIN ---
    # Start: zwei am weitesten entfernte Punkte
    i0, j0 = np.unravel_index(np.argmax(D), D.shape)
    selected = [i0, j0]
    remaining = set(range(n)) - set(selected)

    # iterativ: wähle Punkt mit maximaler minimaler Distanz zu selected
    while len(selected) < n_train:
        rem_list = np.array(list(remaining), dtype=int)
        # min distance to any selected for each remaining
        min_d = np.min(D[rem_list][:, selected], axis=1)
        k = rem_list[np.argmax(min_d)]
        selected.append(int(k))
        remaining.remove(int(k))

    train_idx = np.array(selected, dtype=int)
    val_idx = np.array(sorted(list(remaining)), dtype=int)

    # --- zurück in Originaltypen (DataFrame bleibt DataFrame) ---
    X_train = X.iloc[train_idx] if hasattr(X, "iloc") else Xn[train_idx]
    X_val   = X.iloc[val_idx]   if hasattr(X, "iloc") else Xn[val_idx]
    y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y.reshape(-1, 1)[train_idx].reshape(-1)
    y_val   = y.iloc[val_idx]   if hasattr(y, "iloc") else y.reshape(-1, 1)[val_idx].reshape(-1)

    return X_train, X_val, y_train, y_val

def local_regression_mix(
    X: pd.DataFrame,
    y: pd.Series,
    n_augmented: int,
    k_neighbors: int = 5,
    seed: int = 0
    ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Lokale mischbasierte Datenerweiterung für Regression (vereinfacht).
    - NUR Trainingsdaten!
    - Auswahl lokaler Nachbarn im X-Raum
    - lineare Interpolation von (X, y)
    """
    if n_augmented is None or int(n_augmented) <= 0:
        return X, y

    n_augmented = int(n_augmented)
    k_neighbors = int(k_neighbors)

    rng = np.random.default_rng(seed)

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float).reshape(-1, 1)
    n_samples = X_np.shape[0]

    # Wenn zu wenig Punkte: keine Augmentation
    if n_samples < 2:
        return X, y

    # k begrenzen, damit es immer funktioniert
    k_neighbors = max(1, min(k_neighbors, n_samples - 1))

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X_np)
    neighbors = nn.kneighbors(X_np, return_distance=False)

    X_new = np.empty((n_augmented, X_np.shape[1]), dtype=float)
    y_new = np.empty((n_augmented,), dtype=float)

    for t in range(n_augmented):
        i = rng.integers(0, n_samples)
        neigh_idx = neighbors[i][1:]          # ohne sich selbst
        j = rng.choice(neigh_idx)

        lam = rng.uniform(0.0, 1.0)

        X_new[t] = lam * X_np[i] + (1.0 - lam) * X_np[j]
        y_new[t] = float(lam * y_np[i] + (1.0 - lam) * y_np[j])

    X_aug = pd.concat([X, pd.DataFrame(X_new, columns=X.columns)], ignore_index=True)
    y_aug = pd.concat([y, pd.Series(y_new, name=y.name)], ignore_index=True)

    return X_aug, y_aug

# Train-Test Erstellung mit Kennard Stone
def data():
    """
    Loads data from a specified Excel file, prepares input and output features,
    normalizes the data, and splits it into training, validation, and test sets.

    This function processes a dataframe by segregating input and output
    features, normalizing as needed, and performing a train-test split.
    Additionally, it separates validation data from the training set for model
    evaluation purposes.

    :raises FileNotFoundError: If the specified data file does not exist.
    :raises ValueError: If the dataframe structure does not match the
       expected format or the train-test split ratios are invalid.

    :return: A tuple containing the following elements:
        - `x_train`: Training input data.
        - `y_train`: Training output data.
        - `x_val`: Validation input data.
        - `y_val`: Validation output data.
        - `x_test`: Test input data.
        - `y_test`: Test output data.
    :rtype: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series]
    """
    # DataFrame Laden
    holdout_filename = hpoptimize.test_data
    pool_filename = hpoptimize.train_data

    holdout_df = pd.read_excel(holdout_filename)
    pool_df = pd.read_excel(pool_filename)

    holdout_datain = holdout_df.iloc[:, :10]  # Spalten Input
    holdout_dataout = holdout_df.iloc[:, -3]  # Spalten Output
    pool_datain = pool_df.iloc[:, :10]  # Spalten Input
    pool_dataout = pool_df.iloc[:, -3]  # Spalten Output

    # Normalisieren
    hpoptimize.data_min_x = pool_datain.min().values.astype(float)
    hpoptimize.data_max_x = pool_datain.max().values.astype(float)
    hpoptimize.data_min_y = float(pool_dataout.min())
    hpoptimize.data_max_y = float(pool_dataout.max())

    pool_df_in = (pool_datain - hpoptimize.data_min_x) / (hpoptimize.data_max_x - hpoptimize.data_min_x)
    pool_df_out = (pool_dataout - hpoptimize.data_min_y) / (hpoptimize.data_max_y - hpoptimize.data_min_y)
    holdout_df_in = (holdout_datain - hpoptimize.data_min_x) / (hpoptimize.data_max_x - hpoptimize.data_min_x)
    holdout_df_out = (holdout_dataout - hpoptimize.data_min_y) / (hpoptimize.data_max_y - hpoptimize.data_min_y)

    # SPXY Aufteilung (joint X–Y distances)
    x_train, x_val, y_train, y_val = spxy_train_val_split(
        pool_df_in,
        pool_df_out,
        test_size=0.111111,
        alpha=0.5
    )
    # ============================================================
    # RegMix / lokale mischbasierte Datenerweiterung (NUR Training)
    # ============================================================
    aug = getattr(hpoptimize, "n_augmented", 0)

    # aug kann sein:
    # - 0          -> aus
    # - 0.5        -> +50% von len(x_train)
    # - 500        -> +500 Punkte fix
    if isinstance(aug, float) and 0.0 < aug < 1.0:
        n_aug = int(round(aug * len(x_train)))
    else:
        n_aug = int(aug)

    if n_aug > 0:
        x_train, y_train = local_regression_mix(
            X=x_train,
            y=y_train,
            n_augmented=n_aug,
            k_neighbors=int(getattr(hpoptimize, "k_neighbors", 5)),
            seed=int(hpoptimize.seed)
        )

    x_test, y_test = holdout_df_in, holdout_df_out

    return x_train, y_train, x_val, y_val, x_test, y_test

def unnormalize_x(x_norm):
    return x_norm * (hpoptimize.data_max_x - hpoptimize.data_min_x) + hpoptimize.data_min_x

def unnormalize_y(y_norm):
    return y_norm * (hpoptimize.data_max_y - hpoptimize.data_min_y) + hpoptimize.data_min_y

def save_split_data(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Speichert die Trainings-, Validierungs- und Testdaten in eine separate Excel-Datei
    unabhängig vom Trial, um Nachvollziehbarkeit zu gewährleisten.

    :param x_train: Trainings-Eingabedaten
    :param y_train: Trainings-Zieldaten
    :param x_val: Validierungs-Eingabedaten
    :param y_val: Validierungs-Zieldaten
    :param x_test: Test-Eingabedaten
    :param y_test: Test-Zieldaten
    """
    output_folder = f"Ergebnisse_Teil_3/{hpoptimize.study_name}"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(hpoptimize.paths["data"], f"Split_Daten_{hpoptimize.study_name}.xlsx")

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        # Training
        train_df = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        train_df.to_excel(writer, sheet_name="Train", index=False)

        # Validation
        val_df = pd.concat([x_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
        val_df.to_excel(writer, sheet_name="Validation", index=False)

        # Test
        test_df = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        test_df.to_excel(writer, sheet_name="Test", index=False)

# Erstellt ein Modell und gibt dieses aus
def create_model(trial):
    """
    Creates a Keras sequential model for a given trial. The model's architecture
    and hyperparameters are dynamically generated based on the suggestions provided
    by the trial object. The model includes a flattened layer, multiple hidden layers
    with user-defined activation functions and units, and an output layer. The model
    is compiled using Huber loss, root mean squared error as a metric, and the
    AdamW optimizer configured with dynamically tuned learning rate and weight decay.

    :param trial: An Optuna trial object that suggests hyperparameter values
        for creating the model.
    :type trial: optuna.trial.Trial
    :return: A compiled Keras sequential model.
    :rtype: keras.Sequential
    """
    # Parameter
    n_layers = trial.suggest_int(name="n_Layers", low=1, high=3)
    learning_rate = trial.suggest_float(name="Learning_Rate", low=0.0001, high=1, log=True)
    weight_decay = trial.suggest_float(name="Weight_Decay", low=10e-10, high=10e-3, log=True)

    # Erstelle Model
    model = keras.Sequential()
    model.add(keras.layers.Flatten())

    # Hidden Layer
    for i in range(n_layers):
        act_func = trial.suggest_categorical("act_func_l{}".format(i), ["tanh", "relu"])
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(
            keras.layers.Dense(
                num_hidden,
                activation=act_func,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
            )
        )

    # Ausgangslayer (Anzahl der Ausgangsparameter kann angepasst werden)
    model.add(
        keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    )

    # Model kompilieren
    model.compile(loss=keras.losses.Huber(),
                  metrics=[keras.metrics.RootMeanSquaredError()],
                  optimizer=keras.optimizers.AdamW(
                      learning_rate=learning_rate,
                      weight_decay=weight_decay,
                  )
                  )

    return model


# Nutzt ein Modell und gibt das Ergebniss des Modells aus
def objective(trial):
    """
        Objective function for optimizing a machine learning model using hyperparameter tuning.

        The function orchestrates the creation, evaluation, and storage of a machine learning
        model based on trial parameters. It ensures reproducibility by setting random seeds,
        trains the model on the provided dataset, monitors performance on validation data,
        and computes key metrics such as RMSE and R2 scores. Additionally, it saves the best-performing
        model and generates visualization plots for analysis.

        :param trial: Optuna trial object used for hyperparameter optimization.
        :type trial: optuna.trial.Trial
        :return: Validation RMSE score obtained from the best model.
        :rtype: float
        """
    global current_model  # Reference the global model variable

    # Reproduzierbar machen
    os.environ['PYTHONHASHSEED'] = str(hpoptimize.seed)
    np.random.seed(hpoptimize.seed)
    tf.random.set_seed(hpoptimize.seed)
    tf.keras.utils.set_random_seed(hpoptimize.seed)
    tf.config.experimental.enable_op_determinism()

    # Test und Trainingsdaten einlesen
    x_train, y_train, x_val, y_val, x_test, y_test = data()

    # Erstelle Modell
    model = create_model(trial)
    current_model = model  # Update the global model reference

    # individueller Checkpoint-Pfad pro Trial (liegt im models-Ordner der Study)
    checkpoint_path = os.path.join(
        hpoptimize.paths["models"],
        f"checkpoint_trial_{trial.number}.keras"
    )

    # Parameter Callback
    callback = [
        keras.callbacks.EarlyStopping(
            monitor='val_root_mean_squared_error',
            patience=hpoptimize.patience,
            mode="min"
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_root_mean_squared_error",
            mode="min",
            save_best_only=True,
            verbose=0
        ),
        TFKerasPruningCallback(trial, 'val_root_mean_squared_error')
    ]

    batch_size = trial.suggest_int("Batch_Size", low=1, high=32, log=True)

    # Model fitten
    try:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=hpoptimize.epochs,
            verbose=0,
            shuffle=True,
            callbacks=callback
        )
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} wurde gepruned.")
        trial.set_user_attr("pruned", True)
        raise

    # Bestes Modell aus dem trial-spezifischen Checkpoint laden
    if os.path.exists(checkpoint_path):
        best_model = keras.models.load_model(checkpoint_path)
    else:
        print("Warnung: Modelldatei nicht gefunden! Speichere das letzte Modell stattdessen.")
        best_model = model
        best_model.save(checkpoint_path)

    # Evaluiere bestes Modell
    print("train: ")
    train_predict = best_model.predict(x_train)
    y_train_unnormalized = unnormalize_y(y_train)
    train_predict_unnormalized = unnormalize_y(train_predict)
    train_rmse = root_mean_squared_error(y_train_unnormalized, train_predict_unnormalized)
    train_r2 = r2_score(y_train_unnormalized, train_predict_unnormalized)
    print("val: ")
    val_predict = best_model.predict(x_val)
    y_val_unnormalized = unnormalize_y(y_val)
    val_predict_unnormalized = unnormalize_y(val_predict)
    val_rmse = root_mean_squared_error(y_val_unnormalized, val_predict_unnormalized)
    val_r2 = r2_score(y_val_unnormalized, val_predict_unnormalized)
    print("test: ")
    test_predict = best_model.predict(x_test)
    y_test_unnormalized = unnormalize_y(y_test)
    test_predict_unnormalized = unnormalize_y(test_predict)
    test_rmse = root_mean_squared_error(y_test_unnormalized, test_predict_unnormalized)
    test_r2 = r2_score(y_test_unnormalized, test_predict_unnormalized)

    # ==== Ordner robust anlegen ====
    for key in ["metrics", "plots", "models"]:
        os.makedirs(hpoptimize.paths[key], exist_ok=True)

    # Save to Excel
    save_trial_results_with_dynamic_style(
        trial.number,
        y_train_unnormalized, train_predict_unnormalized,
        y_test_unnormalized, test_predict_unnormalized,
        y_val_unnormalized, val_predict_unnormalized,
        history
    )

    # Speichere Scores für 1 Ausgang

    trial.set_user_attr("train_r2", train_r2)
    trial.set_user_attr("val_r2", val_r2)
    trial.set_user_attr("test_r2", test_r2)
    trial.set_user_attr("train_rmse", train_rmse)
    trial.set_user_attr("val_rmse", val_rmse)
    trial.set_user_attr("test_rmse", test_rmse)

    # Speichern des Models
    model_path = os.path.join(hpoptimize.paths["models"], f"{hpoptimize.study_name}_{trial.number}.keras")
    best_model.save(model_path)

    # Plotte Grafiken für Trial
    plot_graphs(
        history,
        trial.number,
        y_train_unnormalized, train_predict_unnormalized,
        y_val_unnormalized, val_predict_unnormalized,
        y_test_unnormalized, test_predict_unnormalized
    )

    # Metriken speichern
    metrics_df = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [train_rmse, val_rmse, test_rmse],
        'R2': [train_r2, val_r2, test_r2]
    })
    metrics_df.to_csv(os.path.join(hpoptimize.paths["metrics"], f"metrics_{trial.number}.csv"), index=False)

    if np.isnan(val_rmse):
        print(f"Trial {trial.number}: val_rmse ist NaN!")

    return val_rmse


# Global variable to store the study
current_study = None


def save_study_results():
    """
    Save the results of the current optimization study to a directory. This function
    checks if a global variable for the current study exists, and if it does, it saves
    the trials' data as an Excel file in a specific directory named after the
    study name. Creates the directory if it does not already exist for organization
    purposes.

    :raises FileNotFoundError: If the directory path cannot be created.
    :raises PermissionError: If there are not enough permissions to write the file.
    """
    global current_study
    if current_study:
        print("\nSaving study results...")
        study_name = hpoptimize.study_name
        study_path = f"Ergebnisse_Teil_3/{study_name}"
        os.makedirs(study_path, exist_ok=True)  # Ensure the directory exists
        study_file = f"{study_path}/{study_name}.xlsx"
        current_study.trials_dataframe().to_excel(study_file)
        print(f"Study results saved to: {study_file}")

# Register the cleanup function
atexit.register(save_study_results)

def save_model_and_study_on_interrupt(signum, frame):
    """
    Signal handler to save the current model and study on interruption.

    This function is triggered when the program receives a specified signal,
    typically indicating an interruption, such as a keyboard interruption.
    If a model is currently loaded, it saves the model to a specified file and
    persists the study results before gracefully terminating the program.

    :param signum: Signal number associated with the interrupt event.
    :type signum: int
    :param frame: Current stack frame when the signal was received.
    :type frame: FrameType
    :return: None
    """
    global current_model, current_study
    if current_model:
        print("\nProgram interrupted. Saving the current model...")
        current_model.save("interrupted_model.keras")
        print("Model saved as 'interrupted_model.keras'.")
    save_study_results()
    print("Exiting program.")
    exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, save_model_and_study_on_interrupt)

def cleanup_top10(study, trial, base_name):
    """
    Hält nur die Top-10 Trials (nach val_rmse) und löscht alle anderen Dateien.
    """
    # sortiere alle erfolgreichen Trials
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(complete_trials, key=lambda t: t.value)  # kleinster RMSE = bester

    # Top 10 Trials bestimmen
    top10 = sorted_trials[:10]
    top10_ids = {t.number for t in top10}

    # aktuelle Verzeichnisse
    model_dir   = hpoptimize.paths["models"]
    plots_dir   = hpoptimize.paths["plots"]
    metrics_dir = hpoptimize.paths["metrics"]

    # welche Dateien behalten?
    keep_models  = {os.path.join(model_dir, f"{base_name}_{t.number}.keras") for t in top10}
    keep_excels  = {os.path.join(metrics_dir, f"Trial_{t.number}.xlsx") for t in top10}
    keep_metrics = {os.path.join(metrics_dir, f"metrics_{t.number}.csv") for t in top10}
    keep_plots   = {
        os.path.join(plots_dir, f"{t.number}_RMSE.png") for t in top10
    }.union({
        os.path.join(plots_dir, f"{t.number}_Huber_Loss.png") for t in top10
    }).union({
        os.path.join(plots_dir, f"{t.number}_Predict_vs._True.png") for t in top10
    })

    # löschen, was nicht in den keep-Sets ist
    for f in glob.glob(f"{model_dir}/*.keras"):
        if f not in keep_models:
            os.remove(f)
    for f in glob.glob(f"{metrics_dir}/Trial_*.xlsx"):
        if f not in keep_excels:
            os.remove(f)
    for f in glob.glob(f"{metrics_dir}/metrics_*.csv"):
        if f not in keep_metrics:
            os.remove(f)
    for f in glob.glob(f"{plots_dir}/*.png"):
        if f not in keep_plots:
            os.remove(f)

    print(f"Cleanup done → Top 10 Trials: {[t.number for t in top10]}")

# Main
def run_hpo(study_name, seed, patience, epochs, n_trials, train_data, test_data, n_augmented=0, k_neighbors=5):
    """
    Execute a hyperparameter optimization (HPO) workflow using Optuna. The function initializes
    an Optuna study with specified parameters, performs optimization over a defined number
    of trials, and manages cleanup by retaining only the top 10 performing models and their
    associated plots. Intermediate output and results are stored for later analysis.

    :param study_name: The name of the study to be executed and stored.
    :type study_name: str
    :param seed: Seed value used for random operations to ensure reproducibility.
    :type seed: int
    :param patience: Early stopping patience setting for training.
    :type patience: int
    :param epochs: Maximum number of epochs for model training.
    :type epochs: int
    :param n_trials: Number of optimization trials to perform.
    :type n_trials: int
    :param train_data: The training dataset to be used for model training and evaluation.
    :type train_data: Any
    :return: None
    """
    global current_study
    initialize(study_name, seed, patience, epochs, n_trials, train_data, test_data)
    hpoptimize.n_augmented = n_augmented  # kann float (ratio) oder int (absolute) sein
    hpoptimize.k_neighbors = int(k_neighbors)
    # Ergebnisse_Teil_2-Unterordner für bessere Struktur
    base_dir = f"Ergebnisse_Teil_3/{hpoptimize.study_name}"
    hpoptimize.paths = {
        "base": base_dir,
        "models": os.path.join(base_dir, "models"),
        "plots": os.path.join(base_dir, "plots"),
        "metrics": os.path.join(base_dir, "metrics"),
        "data": os.path.join(base_dir, "data")
    }
    for path in hpoptimize.paths.values():
        os.makedirs(path, exist_ok=True)

    # Lade normalisierte Daten
    x_train, y_train, x_val, y_val, x_test, y_test = data()
    save_split_data(x_train, y_train, x_val, y_val, x_test, y_test)

    # Unnormalisierte Daten zum Vergleich speichern
    save_split_data(
        unnormalize_x(x_train),
        unnormalize_y(y_train),
        unnormalize_x(x_val),
        unnormalize_y(y_val),
        unnormalize_x(x_test),
        unnormalize_y(y_test)
    )

    sampler = optuna.samplers.TPESampler(seed=hpoptimize.seed, multivariate=True, n_startup_trials=50)
    pruner = optuna.pruners.MedianPruner(
            n_startup_trials=200,
            n_warmup_steps=300,
            interval_steps=25
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=hpoptimize.study_name,
        storage=f"sqlite:///{os.path.join(base_dir, hpoptimize.study_name + '.sqlite3')}",
        load_if_exists=True
    )

    def per_trial_callback(study, trial):
        cleanup_top10(study, trial, hpoptimize.study_name)

    current_study = study
    study.optimize(objective, n_trials=hpoptimize.n_trials,
                   callbacks=[per_trial_callback])

    save_study_results()

def load_best_value_from_study(study_name: str) -> float:
    base_dir = f"Ergebnisse_Teil_3/{study_name}"
    os.makedirs(base_dir, exist_ok=True)  # stellt sicher, dass der Pfad existiert
    storage = f"sqlite:///{os.path.join(base_dir, study_name + '.sqlite3')}"
    st = optuna.load_study(study_name=study_name, storage=storage)
    return float(st.best_value)

def get_target_rmse_from_study_xlsx(study_name: str,
                                    base_dir: str = "Ergebnisse_Teil_2",
                                    value_col: str = "value",
                                    state_col: str = "state") -> float:
    """
    Liest Ergebnisse_Teil_2/<study_name>/<study_name>.xlsx (Optuna trials_dataframe)
    und gibt den besten (kleinsten) value unter COMPLETE zurück.
    """
    xlsx_path = os.path.join(base_dir, study_name, f"{study_name}.xlsx")
    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"Study-Excel nicht gefunden: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    if value_col not in df.columns:
        raise KeyError(f"Spalte '{value_col}' nicht gefunden in {xlsx_path}. Vorhanden: {list(df.columns)}")
    if state_col not in df.columns:
        # Fallback: manche Exporte nennen es 'state' oder haben es anders; ohne state können wir nicht filtern
        raise KeyError(f"Spalte '{state_col}' nicht gefunden in {xlsx_path}. Vorhanden: {list(df.columns)}")

    df_complete = df[df[state_col].astype(str).str.upper().eq("COMPLETE")].copy()
    df_complete = df_complete[pd.to_numeric(df_complete[value_col], errors="coerce").notna()]

    if df_complete.empty:
        raise ValueError(f"Keine COMPLETE-Trials mit gültigem '{value_col}' in {xlsx_path} gefunden.")

    # RMSE wird minimiert -> bester Wert ist MIN
    return float(df_complete[value_col].min())

if __name__ == '__main__':
    holdout_data = r"Getrennte_Daten/Holdout_fixed_Modell_1.xlsx"
    pool_data    = r"Getrennte_Daten/Pool_Halton_Modell_1.xlsx"

    study_name = "Study_15_10_2025_Halton_Modell_1.2_Halton_SPXY_RegMix_Holdout_seed_0"

    # Beispiel: +100% zusätzliche Trainingspunkte (nach Split!)
    # Wenn du lieber fix willst: z.B. n_augmented=500
    n_augmented = 0.5      # <-- hier einstellen (0.5 = 50%)
    k_neighbors = 5

    run_hpo(
        study_name=study_name,
        seed=0,
        patience=100,
        epochs=1000,
        n_trials=500,
        train_data=pool_data,
        test_data=holdout_data,
        n_augmented=n_augmented,
        k_neighbors=k_neighbors
    )

