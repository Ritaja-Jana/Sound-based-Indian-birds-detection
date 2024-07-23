import numpy as np
import tensorflow as tf
import pickle
import librosa
import os


MODEL_PATH = os.path.join('src', 'model', 'bird_call_model.keras')
LABEL_ENCODER_PATH = os.path.join('src', 'model', 'label_encoder.pkl')

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the label encoder
with open(LABEL_ENCODER_PATH, 'rb') as file:
    label_encoder = pickle.load(file)


def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = extract_features(y, sr)
    return features

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.flatten()

def predict_bird_species(audio_file_path):
    features = preprocess_audio(audio_file_path)
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_species = label_encoder.inverse_transform(predicted_class)
    return predicted_species[0]
