# 🎵 Playlist Curator

A machine learning based playlist recommendation system that analyzes a user's liked songs and recommends new songs based on their musical preferences.

The system combines **machine learning genre classification** with **content-based recommendation using cosine similarity** to generate a personalized playlist.

---

## 📌 Project Overview

Playlist Curator takes a set of songs selected by the user and analyzes their Spotify audio features.

The system works in two main stages:

1. A trained **RBF Support Vector Machine (SVM)** predicts the genres that best match the user's selected songs.
2. **Cosine similarity** compares the audio features of the user's liked songs with other songs in the dataset and ranks the most similar songs.

The result is a personalized playlist containing songs that are musically similar to the user's preferences.

---

## 🎯 Objectives

- Build a machine learning based playlist recommendation system.
- Analyze Spotify audio features to understand musical characteristics.
- Classify songs into selected music genres.
- Recommend songs based on the user's selected songs.
- Compare different machine learning models.
- Select the best performing model using cross-validation and GridSearchCV.
- Provide a simple and interactive web interface using Streamlit.

---

## 🗂️ Dataset

The project uses a Spotify songs dataset containing approximately **114,000 songs** across **114 genres**.

Each genre contains approximately 1,000 songs.

### Selected Genres

For this project, 10 genres were selected:

- Pop
- Rock
- Hip-Hop
- Jazz
- Classical
- Metal
- Country
- Blues
- Reggae
- Electronic

This produces a working dataset of approximately **10,000 songs**.

---

## 🎧 Audio Features

The recommendation and classification system uses the following 10 Spotify audio features:

| Feature | Description |
|---|---|
| Danceability | How suitable a track is for dancing |
| Energy | Perceptual measure of intensity and activity |
| Key | Estimated musical key |
| Loudness | Overall loudness of the track |
| Mode | Major or minor modality |
| Speechiness | Presence of spoken words |
| Acousticness | Confidence that the track is acoustic |
| Liveness | Presence of an audience in the recording |
| Valence | Musical positivity/happiness |
| Tempo | Estimated beats per minute |

The target variable is:

```text
track_genre