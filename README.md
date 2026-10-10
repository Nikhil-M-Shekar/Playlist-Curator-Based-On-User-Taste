# Playlist Curator Based on User Taste

A machine-learning project that recommends songs using their audio features and predicts music genres with a Support Vector Machine (SVM).

## Features

- Select songs you like to build a simple taste profile.
- Recommend similar songs using cosine similarity over scaled audio features.
- Predict song genres using an SVM classifier with an RBF kernel.
- Explore the dataset, compare models, and evaluate classification performance in a Jupyter notebook.
- View recommendations in a Streamlit web app.

## Project Structure

```text
Playlist-Curator-Based-On-User-Taste/
├── app/
│   └── app.py
├── data/
│   └── raw/
│       └── spotify_dataset/
│           └── dataset.csv        # Download/place locally; not committed to Git
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── ml_model.py
│   ├── best_svm.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or a compatible version supported by the packages in `requirements.txt`
- Git
- The Spotify tracks dataset used by this project

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nikhil-M-Shekar/Playlist-Curator-Based-On-User-Taste.git
cd Playlist-Curator-Based-On-User-Taste
```

### 2. Create and activate a virtual environment (recommended)

**Windows — Git Bash**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Windows — Command Prompt**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Setup

The raw dataset is not included in the repository. Place the dataset CSV at:

```text
data/raw/spotify_dataset/dataset.csv
```

The training script expects this path. If you have the original ZIP archive, extract it so the CSV is available at that exact location.

The project uses these ten genres:

- Pop
- Rock
- Hip-hop
- Jazz
- Classical
- Metal
- Country
- Blues
- Reggae
- Electronic

It uses ten audio features: danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, and tempo.

## Run the Project

### Option A — Run the Streamlit application

The repository includes saved model artifacts in `src/`. After installing dependencies and placing the dataset where the application expects it, run the app from the project root:

```bash
streamlit run app/app.py
```

If the app reports that a model file is missing, confirm that these files exist:

```text
src/best_svm.pkl
src/scaler.pkl
src/label_encoder.pkl
```

### Option B — Retrain the SVM model

To train the model and regenerate the saved model files, make sure the dataset CSV is in the expected location, then run from the project root:

```bash
python src/ml_model.py
```

The script uses an 80:20 stratified train/test split, scales the features, and uses `GridSearchCV` with five-fold cross-validation to select an SVM regularization parameter `C` from `[0.1, 1, 10, 100]`. It saves the fitted SVM, scaler, and label encoder in `src/`.

### Option C — Explore the notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_data_exploration.ipynb
```

Run the notebook cells in order to reproduce the exploration, preprocessing, model comparisons, and evaluation visualizations. Ensure the dataset is available at the path used by the notebook.

## How Recommendations Work

1. The user selects songs they like.
2. The app creates a taste profile by averaging the selected songs' scaled audio-feature vectors.
3. Cosine similarity compares that profile with candidate tracks.
4. The selected songs are excluded, and the remaining tracks are ranked by similarity.
5. The SVM's predicted genre can be used to help filter the recommendations when enough candidates are available.

The similarity score represents audio-feature alignment with the selected-song profile; it is **not** a probability that the user will like a song.

## Notes and Limitations

- The classifier predicts only the ten genres used in this project.
- Audio-feature similarity does not guarantee personal preference.
- The raw dataset is excluded from GitHub to avoid committing large data files. Each user must obtain the dataset separately and place it at the path shown above.
- Model artifacts are included for convenience; retraining them requires the dataset.
- If your local folder or dataset structure differs, update the relevant paths in the scripts.

## Repository

[Playlist Curator Based on User Taste](https://github.com/Nikhil-M-Shekar/Playlist-Curator-Based-On-User-Taste)
