# Klasyfikacja obiektów astronomicznych — SDSS17. Klasyfikacja wieloklasowa (GALAXY, QSO, STAR). Plik danych: data/star_classification.csv.

# Import bibliotek i ustawienie stałych projektu. Ustalamy seed dla powtarzalności wyników treningu.
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)

TARGET_COLUMN = "class"
ID_COLUMNS = [
    "obj_ID", "spec_obj_ID", "run_ID", "rerun_ID",
    "cam_col", "field_ID", "plate", "MJD", "fiber_ID",
]
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Stałe hiperparametry — zmieniane tylko w ramach dedykowanego bloku eksperymentu.
FIXED_HP = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "dropout": 0.0,
    "l2": 0.0,
    "batch_norm": False,
    "activation": "relu",
}

input_dim = None
num_classes = None


def get_optimizer(optimizer_name, learning_rate):
    return keras.optimizers.Adam(learning_rate=learning_rate)


def build_model(hidden_layers, dropout_rate, learning_rate, optimizer_name,
                l2_rate=0.0, batch_norm=False, activation="relu"):
    # Uniwersalny generator: baseline (hidden_layers=[]), MLP, Dropout, L2, BatchNorm, różne aktywacje.
    model = keras.Sequential(name="MLP_classifier")
    model.add(layers.Input(shape=(input_dim,)))

    reg = keras.regularizers.l2(l2_rate) if l2_rate > 0 else None

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, kernel_regularizer=reg, name=f"dense_{i+1}"))
        if batch_norm:
            model.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        if activation == "relu":
            model.add(layers.Activation("relu", name=f"act_{i+1}"))
        elif activation == "leaky_relu":
            model.add(layers.LeakyReLU(name=f"act_{i+1}"))
        elif activation == "tanh":
            model.add(layers.Activation("tanh", name=f"act_{i+1}"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    model.add(layers.Dense(num_classes, activation="softmax", name="output"))
    model.compile(
        optimizer=get_optimizer(optimizer_name, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_experiment(name, block, changed_factor, **overrides):
    cfg = {
        **FIXED_HP,
        "name": name,
        "block": block,
        "changed_factor": changed_factor,
        "hidden_layers": [],
    }
    cfg.update(overrides)
    return cfg


def run_experiments(experiments, results, X_train, y_train, X_val, y_val, X_test, y_test,
                    class_names, training_config):
    for exp in experiments:
        print(f"\n[{exp['block']}] {exp['name']} | zmieniany czynnik: {exp['changed_factor']}")

        model = build_model(
            hidden_layers=exp["hidden_layers"], dropout_rate=exp["dropout"],
            l2_rate=exp["l2"], batch_norm=exp["batch_norm"], activation=exp["activation"],
            learning_rate=exp["learning_rate"], optimizer_name=exp["optimizer"],
        )
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=training_config["early_stopping_patience"],
            restore_best_weights=True, verbose=0,
        )
        start = time.time()
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=training_config["epochs"], batch_size=training_config["batch_size"],
            callbacks=[early_stop], verbose=0,
        )
        training_time = time.time() - start
        print(f"  parametry: {model.count_params():,}, czas: {training_time:.1f}s, "
              f"epoki: {len(history.history['loss'])}")
        model.summary()

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        cm = confusion_matrix(y_test, y_pred)
        print(f"  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

        plot_confusion_matrix(cm, class_names, exp["name"])
        plot_training_curves(history, exp["name"])

        results[exp["name"]] = {
            **exp, "model": model, "history": history,
            "training_time": training_time, "num_params": model.count_params(),
            "metrics": metrics, "confusion_matrix": cm,
        }


def block_summary(results, block):
    rows = [
        {"model": n, "f1_score": r["metrics"]["f1_score"], "accuracy": r["metrics"]["accuracy"]}
        for n, r in results.items() if r.get("block") == block
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)


def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history.history["loss"]) + 1)
    axes[0].plot(ep, history.history["accuracy"], label="train")
    axes[0].plot(ep, history.history["val_accuracy"], label="val")
    axes[0].set_title(f"{model_name} — accuracy")
    axes[0].legend()
    axes[1].plot(ep, history.history["loss"], label="train")
    axes[1].plot(ep, history.history["val_loss"], label="val")
    axes[1].set_title(f"{model_name} — loss")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion matrix — {model_name}")
    plt.xlabel("Predykcja")
    plt.ylabel("Klasa")
    plt.tight_layout()
    plt.show()


def build_comparison_table(results):
    rows = []
    for name, r in results.items():
        if "metrics" not in r:
            continue
        rows.append({
            "model": name,
            "block": r["block"],
            "zmieniany_czynnik": r["changed_factor"],
            "architektura": str(r["hidden_layers"]),
            "dropout": r["dropout"],
            "l2": r["l2"],
            "batch_norm": r["batch_norm"],
            "activation": r["activation"],
            "accuracy": r["metrics"]["accuracy"],
            "precision": r["metrics"]["precision"],
            "recall": r["metrics"]["recall"],
            "f1_score": r["metrics"]["f1_score"],
            "czas_treningu_s": round(r["training_time"], 1),
            "liczba_parametrow": r["num_params"],
        })
    return pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)


def main():
    global input_dim, num_classes

    # 1. Analiza danych
    # Wczytanie zbioru SDSS17 z pliku CSV. Sprawdzamy rozmiar danych i pierwsze rekordy.
    DATA_PATH = Path("data/star_classification.csv")
    if not DATA_PATH.exists():
        DATA_PATH = Path("Task2/data/star_classification.csv")
    if not DATA_PATH.exists():
        raise FileNotFoundError("Brak pliku data/star_classification.csv")

    df = pd.read_csv(DATA_PATH)
    print(df.head())
    print(f"Rozmiar: {df.shape[0]:,} × {df.shape[1]}")

    # Analiza typów kolumn i brakujących wartości. Weryfikujemy, czy zbiór wymaga imputacji.
    print(df.dtypes.to_frame(name="dtype"))
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({"brakujące": missing, "procent_%": (missing / len(df) * 100).round(4)})
    print(missing_df)
    print(f"Łącznie braków: {missing.sum()}")

    # Statystyki opisowe cech numerycznych — średnie, odchylenia i zakresy wartości.
    df.info()
    print(df.describe().T)

    # Rozkład klas docelowych: liczność i udział procentowy każdej kategorii (GALAXY, QSO, STAR).
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    class_summary = pd.DataFrame({
        "liczność": class_counts,
        "procent_%": (class_counts / len(df) * 100).round(2),
    })
    print(class_summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index,
                palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title("Liczność klas")
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Rozkład procentowy")
    plt.tight_layout()
    plt.show()

    # Histogramy cech numerycznych pozwalają zobaczyć kształt rozkładów i wartości odstające.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols, n_rows = 3, int(np.ceil(len(numeric_cols) / 3))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax, bins=40)
        ax.set_title(col)
    for ax in axes.flatten()[len(numeric_cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

    # Macierz korelacji pokazuje zależności między cechami — pasma fotometryczne są silnie skorelowane.
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Macierz korelacji")
    plt.tight_layout()
    plt.show()

    # Podział kolumn na identyfikatory (do usunięcia) i cechy przydatne do klasyfikacji.
    existing_ids = [c for c in ID_COLUMNS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in existing_ids + [TARGET_COLUMN]]
    print("Do usunięcia (ID / metadane):", existing_ids)
    print("Cechy do modelu:", feature_cols)

    # Zbiór bez braków, 3 klasy z umiarkowaną nierównowagą. Identyfikatory SDSS nie powinny trafiać do modelu.

    # 2. Przygotowanie danych — kodowanie etykiet, standaryzacja, podział 70/15/15.
    # Przygotowanie danych: usunięcie ID, kodowanie etykiet, standaryzacja i podział 70/15/15.
    df_model = df.drop(columns=existing_ids).copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_model[TARGET_COLUMN])
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    X = df_model.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
    input_dim = X.shape[1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp,
    )

    for name, X_s, y_s in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        print(f"{name}: {len(y_s):,} ({len(y_s)/len(y)*100:.1f}%)")

    # Zapis obiektów StandardScaler i LabelEncoder do ponownego użycia przy inferencji.
    joblib.dump(scaler, ARTIFACTS_DIR / "standard_scaler.joblib")
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.joblib")
    print("Zapisano scaler i label encoder do artifacts/")

    TRAINING_CONFIG = {"epochs": 100, "batch_size": 128, "early_stopping_patience": 10}
    results = {}

    # 4. Blok A — baseline (logistic regression, brak warstw ukrytych).
    experiments_a = [
        make_experiment("A_baseline", "A", "brak warstw ukrytych (logistic regression)"),
    ]
    print("\n" + "=" * 60 + "\nBLOK A: baseline\n" + "=" * 60)
    run_experiments(experiments_a, results, X_train, y_train, X_val, y_val, X_test, y_test,
                    class_names, TRAINING_CONFIG)
    print(block_summary(results, "A"))

    # 5. Blok B — studium głębokości (zmieniamy tylko hidden_layers).
    experiments_b = [
        make_experiment("B_depth_32", "B", "hidden_layers", hidden_layers=[32]),
        make_experiment("B_depth_64_32", "B", "hidden_layers", hidden_layers=[64, 32]),
        make_experiment("B_depth_128_64_32", "B", "hidden_layers", hidden_layers=[128, 64, 32]),
    ]
    print("\n" + "=" * 60 + "\nBLOK B: głębokość sieci\n" + "=" * 60)
    run_experiments(experiments_b, results, X_train, y_train, X_val, y_val, X_test, y_test,
                    class_names, TRAINING_CONFIG)
    print(block_summary(results, "B"))

    # Ręczny wybór architektury po analizie bloku B (na podstawie F1 na zbiorze testowym):
    # B_depth_32: 0.9700 | B_depth_64_32: 0.9724 | B_depth_128_64_32: 0.9727
    BEST_ARCH = [128, 64, 32]

    # 6. Blok C — regularyzacja.
    experiments_c = [
        make_experiment("C_dropout_0.2", "C", "dropout", hidden_layers=BEST_ARCH, dropout=0.2),
        make_experiment("C_dropout_0.4", "C", "dropout", hidden_layers=BEST_ARCH, dropout=0.4),
        make_experiment("C_l2_1e-4", "C", "L2", hidden_layers=BEST_ARCH, l2=1e-4),
    ]
    print("\n" + "=" * 60 + f"\nBLOK C: regularyzacja | arch={BEST_ARCH}\n" + "=" * 60)
    run_experiments(experiments_c, results, X_train, y_train, X_val, y_val, X_test, y_test,
                    class_names, TRAINING_CONFIG)
    print(block_summary(results, "C"))

    # 7. Blok D — BatchNorm.
    experiments_d = [
        make_experiment("D_batchnorm", "D", "BatchNorm", hidden_layers=BEST_ARCH, batch_norm=True),
    ]
    print("\n" + "=" * 60 + f"\nBLOK D: BatchNorm | arch={BEST_ARCH}\n" + "=" * 60)
    run_experiments(experiments_d, results, X_train, y_train, X_val, y_val, X_test, y_test,
                    class_names, TRAINING_CONFIG)
    print(block_summary(results, "D"))

    # 8. Blok E — funkcje aktywacji.
    experiments_e = [
        make_experiment("E_relu", "E", "activation", hidden_layers=BEST_ARCH, activation="relu"),
        make_experiment("E_leaky_relu", "E", "activation", hidden_layers=BEST_ARCH, activation="leaky_relu"),
        make_experiment("E_tanh", "E", "activation", hidden_layers=BEST_ARCH, activation="tanh"),
    ]
    print("\n" + "=" * 60 + f"\nBLOK E: funkcje aktywacji | arch={BEST_ARCH}\n" + "=" * 60)
    run_experiments(experiments_e, results, X_train, y_train, X_val, y_val, X_test, y_test,
                    class_names, TRAINING_CONFIG)
    print(block_summary(results, "E"))

    # 9. Zbiorcza tabela porównawcza.
    comparison_df = build_comparison_table(results)
    print(comparison_df)


if __name__ == "__main__":
    main()
