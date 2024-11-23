# Sound Based Indian Birds Detection

This Indian bird detection project is designed to help bird enthusiasts and researchers identify bird species from audio recordings. Leveraging machine learning and deep learning techniques, this web application provides accurate bird species predictions from uploaded audio files. The model is trained from bird voice files downloaded from [Xeno-canto](https://xeno-canto.org/).

The application is able to detect the following birds:

LABELS = [
    "Asian Koel", "Common Kingfisher", "Common Myna", 
    "Common Tailorbird", "Great Egret",
    "House Sparrow", "House Crow",
    "Human", "Indian Cuckoo", "Indian Peafowl", "Indian Chicken", 
    "Red-vented Bulbul", "Rose-ringed Parakeet", "Rufous Treepie", "Tawny Owl",
    "Indian Spot-billed Duck"
]

## Working of the Model

**1. Feature Extraction:**

- Loads an audio file using librosa.load().
- Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio signal using librosa.feature.mfcc().
- Computes the mean of the MFCCs for each feature, creating a feature vector for each audio sample.


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
- Evaluates the model on the test set and prints the accuracy.


## 🔗 Connect On

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ritaja-jana)
