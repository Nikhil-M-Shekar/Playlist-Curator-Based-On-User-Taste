from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Playlist Curator",
    page_icon="🎵",
    layout="wide"
)


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
# HELPER FOR FINDING DUPLICATE SONGS
# ============================================================

def normalize_song_name(name):

    name = str(name).lower()

    # Remove punctuation
    name = re.sub(r"[^a-z0-9\s]", "", name)

    # Remove extra spaces
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df = df[
        df["track_genre"].isin(GENRES)
    ].copy()

    df = df.dropna(
        subset=[
            "track_id",
            "track_name",
            "artists"
        ] + FEATURE_COLS
    )

    df = df.reset_index(drop=True)

    # Create normalized song and artist names
    # for duplicate detection.
    df["normalized_track"] = (
        df["track_name"]
        .apply(normalize_song_name)
    )

    df["normalized_artist"] = (
        df["artists"]
        .apply(normalize_song_name)
    )

    # Remove duplicate songs.
    #
    # Example:
    # "Let It Snow! Let It Snow! Let It Snow!"
    # and
    # "Let It Snow, Let It Snow, Let It Snow"
    #
    # become the same normalized name.
    df = df.drop_duplicates(
        subset=[
            "normalized_track",
            "normalized_artist"
        ]
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            "Trained model not found.\n\n"
            "Run this command from the project root:\n\n"
            "`python src/ml_model.py`"
        )

        st.stop()

    if not SCALER_PATH.exists():

        st.error(
            "Scaler not found.\n\n"
            "Run:\n\n"
            "`python src/ml_model.py`"
        )

        st.stop()

    if not ENCODER_PATH.exists():

        st.error(
            "Label encoder not found.\n\n"
            "Run:\n\n"
            "`python src/ml_model.py`"
        )

        st.stop()

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder


# ============================================================
# PREPARE FEATURES FOR COSINE SIMILARITY
# ============================================================

@st.cache_data
def prepare_features(df):

    scaler = StandardScaler()

    feature_matrix = scaler.fit_transform(
        df[FEATURE_COLS]
    )

    return feature_matrix


# ============================================================
# PREDICT GENRES OF LIKED SONGS
# ============================================================

def predict_liked_genres(
    df,
    liked_track_ids,
    model,
    scaler,
    encoder
):

    liked_songs = df[
        df["track_id"].isin(liked_track_ids)
    ].copy()

    if liked_songs.empty:

        return []

    liked_features = liked_songs[
        FEATURE_COLS
    ]

    liked_features_scaled = scaler.transform(
        liked_features
    )

    predictions = model.predict(
        liked_features_scaled
    )

    predicted_genres = encoder.inverse_transform(
        predictions
    )

    return list(set(predicted_genres))


# ============================================================
# RECOMMEND SONGS
# ============================================================

def recommend_songs(
    df,
    feature_matrix,
    liked_track_ids,
    predicted_genres,
    number_of_recommendations
):

    liked_positions = df.index[
        df["track_id"].isin(liked_track_ids)
    ].tolist()

    if len(liked_positions) == 0:

        return pd.DataFrame()

    # Create average user taste profile
    user_profile = feature_matrix[
        liked_positions
    ].mean(axis=0)

    # Calculate cosine similarity
    similarities = cosine_similarity(
        [user_profile],
        feature_matrix
    )[0]

    results = df.copy()

    results["similarity"] = similarities

    # Don't recommend songs already selected
    results = results[
        ~results["track_id"].isin(liked_track_ids)
    ]

    # Prefer genres predicted by SVM
    if predicted_genres:

        genre_results = results[
            results["track_genre"].isin(
                predicted_genres
            )
        ].copy()

        if len(genre_results) >= number_of_recommendations:

            results = genre_results

    # Extra safety:
    # remove duplicate songs before ranking.
    results = results.drop_duplicates(
        subset=[
            "normalized_track",
            "normalized_artist"
        ]
    )

    # Sort by similarity
    results = results.sort_values(
        "similarity",
        ascending=False
    )

    return results.head(
        number_of_recommendations
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

df = load_data()

feature_matrix = prepare_features(df)

model, model_scaler, encoder = load_model()


# ============================================================
# CREATE SONG SELECTION OPTIONS
# ============================================================

df["display_name"] = (
    df["track_name"].astype(str)
    + " — "
    + df["artists"].astype(str)
)

display_options = df[
    [
        "track_id",
        "display_name"
    ]
].copy()

label_map = dict(
    zip(
        display_options["track_id"],
        display_options["display_name"]
    )
)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎵 Playlist Curator")

st.subheader(
    "Discover songs based on your musical taste"
)

st.write(
    """
    Select songs that you already like and the Playlist Curator
    will analyze their audio features and use a trained machine
    learning model to create a personalized playlist.
    """
)

st.divider()


st.markdown(
    "### ❤️ Select songs you like"
)

selected_tracks = st.multiselect(
    "Choose up to 10 songs:",
    options=display_options[
        "track_id"
    ].tolist(),
    format_func=lambda track_id:
        label_map[track_id],
    max_selections=10
)


number_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


generate = st.button(
    "🎧 Generate Playlist",
    type="primary"
)


# ============================================================
# GENERATE PLAYLIST
# ============================================================

if generate:

    if len(selected_tracks) == 0:

        st.warning(
            "Please select at least one song first."
        )

    else:

        # Predict genres
        predicted_genres = predict_liked_genres(
            df,
            selected_tracks,
            model,
            model_scaler,
            encoder
        )

        # Generate recommendations
        recommendations = recommend_songs(
            df,
            feature_matrix,
            selected_tracks,
            predicted_genres,
            number_of_recommendations
        )
        st.divider()
        st.header(
            "🎶 Your Recommended Playlist"
        )
        st.write(
            "The trained RBF SVM classified your selected "
            "songs into the following genre(s):"
        )
        if predicted_genres:

            st.info(
                ", ".join(
                    genre.title()
                    for genre in predicted_genres
                )
            )
        st.write(
            "Recommendations are then ranked using "
            "cosine similarity between the audio features "
            "of your liked songs and candidate songs."
        )
        st.divider()
        if recommendations.empty:
            st.warning(
                "No recommendations could be generated."
            )
        else:
            for rank, (_, song) in enumerate(
                recommendations.iterrows(),
                start=1
            ):
                st.markdown(
                    f"### {rank}. "
                    f"{song['track_name']}"
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(
                        f"**Artist:** "
                        f"{song['artists']}"
                    )
                with col2:
                    st.write(
                        f"**Genre:** "
                        f"{song['track_genre']}"
                    )
                with col3:
                    st.write(
                        f"**Similarity:** "
                        f"{song['similarity']:.3f}"
                    )
                st.divider()
st.caption(
    "Playlist Curator uses a GridSearchCV-selected "
    "RBF SVM classifier and cosine similarity across "
    "10 Spotify audio features."
)