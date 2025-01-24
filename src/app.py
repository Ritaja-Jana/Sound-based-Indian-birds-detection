import os
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import logging

app = Flask(_name_)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define constants
LABEL_ENCODER_PATH = os.path.join('src', 'model', 'fcnn_label_encoder.pkl')
MODEL_PATH = os.path.join('src', 'model', 'fcnn_bird_call_model.keras')


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}
TEMPERATURE = 0.25  # Adjust this value for best performance

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YAMNet model and fine-tuned model
try:
    yamnet_model = hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as file:
        label_encoder = pickle.load(file)
except Exception as e:
    logging.error(f"Error loading model or label encoder: {e}")
    raise


def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def save_file(file, filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    return file_path

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Function to preprocess audio and extract embeddings
def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # YAMNet expects 16 kHz audio
        scores, embeddings, spectrogram = yamnet_model(audio)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        # Normalize embeddings
        normalized_embedding = (embedding_mean - np.mean(embedding_mean)) / np.std(embedding_mean)
        return normalized_embedding
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/live_process', methods=['POST'])
def live_process():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio data provided"}), 400

    audio_data = request.files['audio_data']
    if not audio_data.filename or not allowed_file(audio_data.filename):
        return jsonify({"error": "Invalid file format"}), 400

    file_path = save_file(audio_data, 'live_audio.wav')
    features = preprocess_audio(file_path)
    
    # Remove the uploaded file after processing
    remove_file(file_path)

    # Check if features are None
    if features is None:
        logging.error("No features extracted.")
        return jsonify({"error": "No bird detected"}), 200
    
    # Reshape features to match expected model input
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    # Predict using the fine-tuned model
    logits = model.predict(features)

    # Method 1: Softmax (Standard Probability Calculation)
    probabilities_sof = tf.nn.softmax(logits)
    #print(f"Softmax Probabilities: {probabilities_sof.numpy()}")

    # Extract the predicted class and its probability from Softmax (Method 1)
    predicted_class = np.argmax(probabilities_sof, axis=1)
    best_probability = probabilities_sof.numpy()[0][predicted_class[0]] * 100

    print(f"Predicted class: {predicted_class}, Probability: {best_probability:.2f}%")

    # Check probability against the threshold
    threshold = N  # Adjust this threshold as needed
    if best_probability < threshold:
        predicted_label = "No bird detected"
    else:
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]

    result = {
        "probability": f"{best_probability:.2f}%" if predicted_label not in ("Human", "Noise") else "5%",
        "bird_name": predicted_label,
        "description": None,
        "image_url": f"/static/images/birds/{predicted_label}.jpg"
    }

    logging.info(f"Detected bird: {predicted_label} with probability: {best_probability:.2f}%")
    
    return jsonify(result)


if _name_ == '_main_':
    app.run(debug=True)
