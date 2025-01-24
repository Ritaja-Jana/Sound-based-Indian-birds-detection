import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import librosa
from tqdm import tqdm
import pickle
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Define paths and constants
LABEL_ENCODER_PATH = os.path.join('src', 'model', 'fcnn_label_encoder.pkl')
BIRD_MODEL_PATH = os.path.join('src', 'model', 'fcnn_bird_call_model.keras')
DATA_PATH = os.path.join('data', 'xeno-canto-dataset')
LABELS = [
    "Asian Koel", "Common Kingfisher", "Common Myna", 
    "Common Tailorbird", "Great Egret", "Red-whiskered Bulbul",
    "House Sparrow", "House Crow",
    "Human", "Indian Cuckoo", "Indian Peafowl", "Indian Chicken", 
    "Indian Spot-billed Duck", "Noise", "Rose-ringed Parakeet",
    "Rufous Treepie", "Rock Pigeon", "Tawny Owl"
]

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")

# Function to preprocess audio and extract embeddings
def extract_embeddings(file_path):
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

# Load and preprocess the dataset
def load_data():
    X, y = [], []
    for label in LABELS:
        folder_path = os.path.join(DATA_PATH, label)
        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}", unit="file"):
            file_path = os.path.join(folder_path, file_name)
            features = extract_embeddings(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
    np.savez('src/data/embeddings_labels.npz', X=X, y=y)
    print("\nThe labels & embeddings have been saved.\n")
    return np.array(X), np.array(y)

# Build the FCNN model
def build_fcnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(1e-4)), # Rectified Linear Unit (-ve -> 0, +ve -> same)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if _name_ == '_main_':
    # Load data and extract features
    X, y = load_data()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Build and train model
    model = build_fcnn_model((X_train.shape[1],), len(LABELS))

    # Callbacks
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Train the model with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[lr_scheduler, early_stopping]
    )

    # Save model and label encoder
    model.save(BIRD_MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'wb') as file:
        pickle.dump(label_encoder, file)

    # Save training history for later use
    with open('src/model/training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    # Classification report
    y_val_pred_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    print(classification_report(y_val, y_val_pred, target_names=LABELS)) 
