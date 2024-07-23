# Sound Based Indian Birds Detection

This Indian bird detection project is designed to help bird enthusiasts and researchers identify bird species from audio recordings. Leveraging machine learning and deep learning techniques, this web application provides accurate bird species predictions from uploaded audio files. The model is trained from bird voice files downloaded from [Xeno-canto](https://xeno-canto.org/).

The application is able to detect the following birds:

1. Oriental Magpie-Robin
2. Asian Koel
3. Common Tailorbird
4. Rufous Treepie
5. Black-hooded Oriole
6. White-cheeked Barbet
7. Ashy Prinia
8. Puff-throated Babbler
9. White-throated Kingfisher
10. Red-vented Bulbul
11. Jungle Babbler
12. Common Hawk-Cuckoo
13. Indian Scimitar Babbler
14. Red-whiskered Bulbul
15. Red-wattled Lapwing
16. Common Iora
17. Purple Sunbird
18. Greater Coucal
19. Blyth's Reed Warbler
20. Orange-headed Thrush
21. House Crow
22. Greater Racket-tailed Drongo
23. Malabar Whistling Thrush

## Setup Instructions

### 1. Clone the repository:

```bash
git clone https://github.com/Ritaja-Jana/Sound-based-Indian-birds-detection.git

cd sound-based-Indian-birds-detection
```

### 2. Install dependencies:

Ensure you have Python and pip installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Download Audio Files (Optional):

Download sounds of all the 23 birds by running the following command:

```bash
python data/download_sounds.py
```

### 5. Core Functionality (core.py):

It contains functions for preprocessing audio and predicting bird species using a trained model. This is the core logic behind the predictions. Run the below command:

```bash
python src/core.py
```

### 6. Web Application (app.py):

It provides a web interface for users to upload audio files, processes these files, and displays the prediction results along with relevant bird descriptions and images.

```bash
python src/app.py
```

### 7. Open your web browser and go to http://localhost:5000 to use the application.

## Usage

1. Choose the audio file of any of the 23 birds mentioned above.

2. Click on "Upload" to see what bird the sound belongs to.

*Note: The model may make mistakes in identifying birds from sounds.*

## Working of the Model
### Data Preprocessing (data_preprocessing.py)

**1. Feature Extraction:**

- Loads an audio file using librosa.load().
- Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio signal using librosa.feature.mfcc().
- Computes the mean of the MFCCs for each feature, creating a feature vector for each audio sample.

**2. Processing Directory:**

- Iterates through each subdirectory (representing different bird species) and processes each .mp3 file.
- For each file, it extracts features using extract_features() and associates these features with the label (species).
- Saves the feature vectors and labels to `features.npy` and `labels.npy` respectively.
- Serializes the LabelEncoder to `label_encoder.pkl` to encode the bird species labels into numeric format.

### Model Training (model_training.py)

**1. Load Data:**

- Loads the saved feature vectors and labels from `features.npy` and `labels.npy`.
- Deserializes the LabelEncoder to decode the labels.
- Converts the numeric labels into one-hot encoded vectors using `to_categorical()`.

**2. Build Model:**

- Defines a Sequential neural network model using Keras.
- Includes Dense layers with ReLU activation functions, BatchNormalization, and Dropout layers to prevent overfitting.
- The final layer is a Dense layer with a softmax activation function for classification.
- Compiles the model using the Adam optimizer and categorical crossentropy loss function.

**3. Training and Evaluation**

- Loads the data and label encoder.
- Splits the data into training and testing sets using train_test_split().
- Defines the input shape and number of output classes for the model.
- Builds and trains the model using `model.fit()`.
- Saves the trained model to `birds_call_model.keras`.
- Evaluates the model on the test set and prints the accuracy.


## 🔗 Connect On

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ritaja-jana)
