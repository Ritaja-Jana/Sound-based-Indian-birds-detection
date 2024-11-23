import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import librosa
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
LABEL_ENCODER_PATH = os.path.join('src', 'model', 'simple_label_encoder.pkl')
BIRD_MODEL_PATH = os.path.join('src', 'model', 'simple_bird_call_model.keras')
DATA_PATH = os.path.join('data', 'xeno-canto-dataset')
LABELS = [
    "Asian Koel", "Common Kingfisher", "Common Myna", 
    "Common Tailorbird", "House Sparrow", "House Crow",
    "Human", "Indian Cuckoo", "Indian Peafowl", "Indian Chicken", 
    "Red-vented Bulbul", "Rose-ringed Parakeet", "Rufous Treepie", "Tawny Owl",
    "Common Sandpiper", "Indian Spot-billed Duck"
]
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

def load_data():
    print("\nThe data is being processed...\n")
    X, y = [], []
    for label in LABELS:
        folder_path = os.path.join(DATA_PATH, label)
        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}", unit="file"):
            file_path = os.path.join(folder_path, file_name)
            features = preprocess_audio(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
    print("\nData Processing is completed!\n")
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape), 
        tf.keras.layers.Dense(1024, activation='relu'),       
        tf.keras.layers.Dense(512, activation='relu'), 
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("\nModel building successful!\n")
    return model


if __name__ == '__main__':
    X, y = load_data()
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model((X_train.shape[1],))
    print("\nThe model is being trained...\n")
    model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_val, y_val))

    # Save the model and label encoder 
    model.save(BIRD_MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'wb') as file:
        pickle.dump(label_encoder, file)

    logging.info("Model training completed. The model is saved to given location.")

    # Generate classification report on validation data
    print("\nGenerating classification report on validation data...\n")
    y_val_pred_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)

    print(classification_report(y_val, y_val_pred, target_names=LABELS))

    logging.info("\nThe model summary is: ")
    model.summary()
