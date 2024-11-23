import os
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import librosa
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define constants
MODEL_PATH = os.path.join('src', 'model', 'simple_bird_call_model.keras')
LABEL_ENCODER_PATH = os.path.join('src', 'model', 'simple_label_encoder.pkl')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}
TEMPERATURE = 0.25 # Adjust this value for best performance

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and label encoder
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as file:
        label_encoder = pickle.load(file)
except Exception as e:
    logging.error(f"Error loading model or label encoder: {e}")
    raise

# Bird descriptions dictionary
bird_descriptions = {
    "Indian Peafowl": "A large, colorful bird...",
    "House Sparrow": "A small, adaptable bird...",
    "Common Myna": "A medium-sized bird...",
    "Asian Koel": "A melodious bird...",
}

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def save_file(file, filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    return file_path

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

SAMPLE_RATE = 22050
NUM_MFCC = 100

def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC)
        return np.mean(mfccs, axis=1)  # Return mean MFCCs
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
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
    
    # Check if features are None
    if features is None:
        logging.error("No features extracted.")
        return "No bird detected", 0.0
    
    # Reshape features to match expected model input
    if features.shape != (NUM_MFCC,):
        logging.error(f"Feature shape mismatch: got {features.shape}, expected {(NUM_MFCC,)}")
        return "No bird detected", 0.0
    
    #logging.info(f"Adjusted feature shape: {features.shape}, Expected shape: {model.input_shape[1:]}")
    
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    
    # Proceed with prediction
    logits = model.predict(features)
    
    # Temperature scaling and softmax
    scaled_logits = logits / TEMPERATURE
    probabilities = tf.nn.softmax(scaled_logits)
    
    predicted_class = np.argmax(probabilities, axis=1)
    best_probability = probabilities.numpy()[0][predicted_class[0]] * 100

    # Check probability against the threshold
    threshold = 50 # Adjust this threshold as needed
    if best_probability < threshold:
        predicted_label = "No bird detected"
    else:
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]
        
    if predicted_label == "Human":
        best_probability = -1

    result = {
        "probability": f"{best_probability:.2f}%",
        "bird_name": predicted_label,
        "description": bird_descriptions.get(predicted_label, "Description not available."),
        "image_url": f"/static/images/birds/{predicted_label}.jpg"
    }
    logging.info(f"Detected bird: {predicted_label} with probability: {best_probability:.2f}%")
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

