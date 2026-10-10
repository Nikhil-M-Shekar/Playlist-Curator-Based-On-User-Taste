from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# SETTINGS
# ============================================================

GENRES = [
    "pop",
    "rock",
    "hip-hop",
    "jazz",
    "classical",
    "metal",
    "country",
    "blues",
    "reggae",
    "electronic"
]

FEATURE_COLS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "liveness",
    "valence",
    "tempo"
]

LABEL_COL = "track_genre"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "spotify_dataset"
    / "dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "src"
    / "best_svm.pkl"
)

SCALER_PATH = (
    BASE_DIR
    / "src"
    / "scaler.pkl"
)

ENCODER_PATH = (
    BASE_DIR
    / "src"
    / "label_encoder.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Keep only our 10 selected genres
    df = df[
        df[LABEL_COL].isin(GENRES)
    ].copy()

    # Remove rows with missing values in required columns
    df = df.dropna(
        subset=FEATURE_COLS + [LABEL_COL]
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("Loading dataset...")

    df = load_data()

    print("Dataset shape:", df.shape)

    # Features
    X = df[FEATURE_COLS]

    # Labels
    y = df[LABEL_COL]

    # Encode genre names into numbers
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    print("\nGenre mapping:")

    for number, genre in enumerate(encoder.classes_):
        print(number, "=", genre)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining samples:", X_train.shape)
    print("Testing samples:", X_test.shape)

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # GridSearchCV
    # --------------------------------------------------------

    print("\nRunning GridSearchCV...")

    param_grid = {
        "C": [0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        estimator=SVC(
            kernel="rbf"
        ),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = grid.best_estimator_

    print("\n==============================")
    print("GRID SEARCH RESULTS")
    print("==============================")

    print(
        "Best Parameters:",
        grid.best_params_
    )

    print(
        "Best CV Accuracy:",
        grid.best_score_
    )

    # --------------------------------------------------------
    # Test accuracy
    # --------------------------------------------------------

    y_pred = best_model.predict(
        X_test_scaled
    )

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        "\nTest Accuracy:",
        test_accuracy
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_
        )
    )

    # --------------------------------------------------------
    # Save everything needed by the app
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

    print("\n==============================")
    print("MODEL SAVED")
    print("==============================")

    print("Model:", MODEL_PATH)
    print("Scaler:", SCALER_PATH)
    print("Encoder:", ENCODER_PATH)

    return best_model, scaler, encoder


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()