import os
import numpy as np
import pickle
from flask import Flask, request, render_template
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import librosa

app = Flask(__name__)


# Load the model and label encoder
MODEL_PATH = os.path.join('src', 'model', 'bird_call_model.keras')
LABEL_ENCODER_PATH = os.path.join('src', 'model', 'label_encoder.pkl')

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the label encoder
with open(LABEL_ENCODER_PATH, 'rb') as file:
    label_encoder = pickle.load(file)

# Bird descriptions dictionary
bird_descriptions = {
    "Oriental_Magpie-Robin": "The Oriental Magpie-Robin is a small, insectivorous bird known for its melodious song. It is commonly found in gardens, woodlands, and open forests in South and Southeast Asia.",
    "Asian_Koel": "The Asian Koel is a member of the cuckoo order of birds, known for its distinct call and glossy black plumage. It is found in forests, urban areas, and gardens across South and Southeast Asia.",
    "Common_Tailorbird": "The Common Tailorbird is recognized by its distinctive tail and its ability to sew leaves to build nests. It is commonly found in gardens, forests, and shrubby areas in South and Southeast Asia.",
    "Rufous_Treepie": "The Rufous Treepie is a tree-dwelling bird with rufous and black plumage. It is commonly found in forests and woodlands across the Indian subcontinent.",
    "Black-hooded_Oriole": "The Black-hooded Oriole is known for its bright yellow body and black head. It inhabits dense forests and woodlands of the Indian subcontinent and Southeast Asia.",
    "White-cheeked_Barbet": "The White-cheeked Barbet is known for its vibrant green color and distinctive call. It inhabits forests, wooded areas, and gardens in the Indian subcontinent.",
    "Ashy_Prinia": "The Ashy Prinia is a small, active bird with a distinctive tail. It is commonly found in scrublands, grasslands, and open areas in the Indian subcontinent.",
    "Puff-throated_Babbler": "The Puff-throated Babbler is characterized by its puffy throat and melodious calls. It is found in dense forests and woodlands of the Indian subcontinent.",
    "White-throated_Kingfisher": "The White-throated Kingfisher is known for its striking blue plumage and white throat. It inhabits open woodlands, mangroves, and near water bodies in South Asia.",
    "Red-vented_Bulbul": "The Red-vented Bulbul is a common and noisy bird with a red patch under its tail. It is found in a variety of habitats including gardens, forests, and urban areas across South Asia.",
    "Jungle_Babbler": "The Jungle Babbler is a social bird with a distinctive call and brown plumage. It is commonly found in open woodlands, scrublands, and urban areas in South Asia.",
    "Common_Hawk-Cuckoo": "The Common Hawk-Cuckoo is recognized by its call and grey-brown plumage. It is a brood parasite found in forests and woodlands of South Asia.",
    "Indian_Scimitar_Babbler": "The Indian Scimitar Babbler is known for its long, curved bill and melodious call. It inhabits dense forests and scrublands of the Indian subcontinent.",
    "Red-whiskered_Bulbul": "The Red-whiskered Bulbul is a small, active bird with a distinctive red patch on its cheeks. It is found in gardens, forests, and urban areas across South Asia.",
    "Red-wattled_Lapwing": "The Red-wattled Lapwing is known for its loud calls and distinctive red wattles. It is commonly found in open fields, wetlands, and grasslands in South Asia.",
    "Common_Iora": "The Common Iora is a small, colorful bird with a striking yellow and green plumage. It is found in forests, gardens, and open areas of South Asia.",
    "Purple_Sunbird": "The Purple Sunbird is recognized by its iridescent purple plumage. It is commonly found in gardens, forests, and scrublands in South and Southeast Asia.",
    "Greater_Coucal": "The Greater Coucal is known for its large size and deep call. It inhabits forests, open woodlands, and gardens in South Asia.",
    "Blyth's_Reed_Warbler": "Blyth's Reed Warbler is a small, secretive bird with a cryptic plumage. It is found in reed beds and marshy areas across South and Southeast Asia.",
    "Orange-headed_Thrush": "The Orange-headed Thrush is recognizable by its bright orange head and breast. It is found in forests, gardens, and wooded areas of the Indian subcontinent.",
    "House_Crow": "The House Crow is a highly adaptable and intelligent bird with black plumage. It is commonly found in urban areas, villages, and open spaces across South Asia.",
    "Greater_Racket-tailed_Drongo": "The Greater Racket-tailed Drongo is known for its striking tail feathers and aggressive behavior. It is found in forests and woodlands of South and Southeast Asia.",
    "Malabar_Whistling_Thrush": "The Malabar Whistling Thrush is recognized by its beautiful blue plumage and melodious calls. It is found in forested areas of the Western Ghats in India."
}


def preprocess_audio(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs, axis=1)
    
    return mfccs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result="No file selected")
        
        if file and file.filename.endswith('.mp3'):
            # Save the uploaded file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the audio file
            features = preprocess_audio(file_path)
            features = np.expand_dims(features, axis=0)

            # Predict using the trained model
            predictions = model.predict(features)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]

            # Get prediction percentage of the top prediction
            best_probability = predictions[0][predicted_class][0] * 100

            # Create a formatted result string
            result = f"There is <span class='probability'>{best_probability:.2f}%</span> probability that the bird is <span class='bird-name'>{predicted_label}</span>."

            # Get image URL
            image_url = f"/static/images/birds/{predicted_label}.jpg"

            # Get bird description
            bird_description = bird_descriptions.get(predicted_label, "Description not available.")

            # Remove the uploaded file
            os.remove(file_path)

            return render_template('index.html', result=result, image_url=image_url, bird_description=bird_description)
        
        return render_template('index.html', result="Invalid file format. Please upload an MP3 file.")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
