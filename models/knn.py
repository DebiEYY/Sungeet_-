# Standard library imports
import warnings

# Third-party imports for data manipulation and analysis
import numpy as np
import pandas as pd
from collections import Counter

# Multi processing library
import concurrent.futures

# Audio processing library
import librosa
import librosa.display

# Machine learning preprocessing and model selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Deep learning libraries
from tensorflow.keras.optimizers import Adam

# Statistical distributions for randomized search
from scipy.stats import randint

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Save the model
import joblib

# My file
from converter import midi_to_wav

def split_audio(audio_file, segment_length=3):
    path = midi_to_wav(audio_file, audio_file)
    # Load the audio file
    y, sr = librosa.load(path, sr=None)

    # Calculate segment length in samples
    segment_samples = int(segment_length * sr)

    # Calculate total number of segments
    num_segments = 10

    # Split the audio into segments of specified length
    segments = []
    for i in range(num_segments):
        start_sample = i * segment_samples
        end_sample = (i + 1) * segment_samples
        segment = y[start_sample:end_sample]
        segments.append(segment)

    return segments


def Extract_features(y, sr=22050):
    # Load the audio file
    # y, sr = librosa.load(audio_path, sr=None)

    # Extract features
    features = {
        # 'filename': audio_path,
        'length': len(y),
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rms_mean': np.mean(librosa.feature.rms(y=y)),
        'rms_var': np.var(librosa.feature.rms(y=y)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
        'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y)),
        'harmony_mean': np.mean(librosa.effects.harmonic(y)),
        'harmony_var': np.var(librosa.effects.harmonic(y)),
        'perceptr_mean': np.mean(librosa.effects.percussive(y)),
        'perceptr_var': np.var(librosa.effects.percussive(y)),
        'tempo': librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
    }

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i - 1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i - 1])

    # Create DataFrame
    df = pd.DataFrame([features])
    return df


def preprocessing(audio):
    segments = split_audio(audio)

    audio_df = []

    for segment in segments:
        audio_df.append(Extract_features(segment))

    return audio_df


def read_csv():
    df = pd.read_csv(
        "C:\\Users\\debie\\Desktop\\Studies\\project\\knn_genre_classification\\archive\\Data\\features_3_sec.csv")
    # df.drop(0, axis = 1, inplace=True)
    df.drop('filename', axis=1, inplace=True)
    return df


def fit(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # חלוקת הדטה לקבוצות אימון ובדיקה
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # קידוד התוויות
    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(y_train)
    y_test = labelencoder.transform(y_test)

    # סקיילינג של הדטה
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # הגדרת פרמטרים לחיפוש רנדומלי
    param_grid = {
        'n_neighbors': randint(1, 15),  # Number of neighbors
        'weights': ['uniform', 'distance'],  # Weight function
        'p': [1, 2]  # Power parameter for the Minkowski distance metric
    }

    # יצירת המודל
    knn = KNeighborsClassifier()

    # חיפוש רנדומלי
    random_search_knn = RandomizedSearchCV(
        knn, param_distributions=param_grid, n_iter=50, cv=5, random_state=42
    )

    # אימון המודל
    random_search_knn.fit(X_train, y_train)

    # הערכת המודל על סט הבדיקה
    best_knn = random_search_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    test_accuracy_knn = accuracy_score(y_test, y_pred_knn)

    # הערכת המודל על סט האימון
    y_train_pred_knn = best_knn.predict(X_train)
    train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)

    # רוצה לראות  בעיניים את ההערכות
    print("Train KNN Accuracy:", train_accuracy_knn)
    print("Test KNN Accuracy:", test_accuracy_knn)

    # הדפסת התוצאות של הסיווג
    print("Unique predictions in test set:", set(y_pred_knn))

    # החזרה של המודל המאומן, של המקודד ושל הסקלר
    return random_search_knn, labelencoder, scaler


def predict_new_song(model, labelencoder, scaler, new_song):
    # Transform and scale new song features
    new_song_transformed = scaler.transform(new_song)

    # Predict genre
    y_pred = model.predict(new_song_transformed)

    # Decode label
    y_pred_decoded = labelencoder.inverse_transform(y_pred)

    y_pred_decoded = str(y_pred_decoded)
    y_pred_decoded = y_pred_decoded.replace("['", '').replace("']", '')
    print(y_pred_decoded)

    return y_pred_decoded


def save_model():
    df = read_csv()

    # Fit the model with dataset
    model, labelencoder, scaler = fit(df)

    joblib.dump((model, labelencoder, scaler), 'knn_model.pkl')


def get_genre_by_knn(path):
    audio = path

    # Fit the model with dataset
    model, labelencoder, scaler = joblib.load('knn_model.pkl')

    song_datas_segments = preprocessing(audio)

    # מספר השירים לנבא להם את הז'אנר
    num_songs = len(song_datas_segments)

    # הרצת הפונקציה predict_genre במקביל על כל השירים
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_songs) as executor:
        futures = [
            executor.submit(predict_new_song, model, labelencoder, scaler, segment)
            for segment in song_datas_segments
        ]

        # איסוף התוצאות של הניבויים
        genres = []
        for future in concurrent.futures.as_completed(futures):
            try:
                genre = future.result()
                genres.append(genre)
            except Exception as e:
                print(f"Prediction generated an exception: {e}")

    genre_count = Counter(genre for genre in genres)
    most_common_genre, count = genre_count.most_common(1)[0]
    print(most_common_genre)

    return most_common_genre


get_genre_by_knn(r"C:\Users\debie\Desktop\Studies\project\genrate_music_rnn\example.midi")