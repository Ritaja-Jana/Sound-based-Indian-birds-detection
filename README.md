# Indian Birds Detection

This project aims to classify emails as spam or not spam using machine learning techniques. The classification is based on the content and metadata of the email. The model used in this project is a **RandomForestClassifier** trained on the Spambase Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase).

## Setup Instructions

### 1. Clone the repository:

```bash
git clone https://github.com/Ritaja-Jana/Email-Spam-Classification.git

cd email-spam-classification
```

### 2. Install dependencies:

Ensure you have Python and pip installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Preprocess the Raw Data:

Convert the raw data into processed data by the following command:

```bash
python src/data_preprocessing.py
```

### 5. Train the ML Model

Train the Model on the processed data using the command:

```bash
python src/model_training.py
```

### 6. Run the Flask application:

Execute the following command to start the Flask web server:

```bash
python src/app.py
```

### 7. Open your web browser and go to http://localhost:5000 to use the application.

## Model Training

1. The model is trained using RandomForestClassifier from scikit-learn.

2. Training involves preprocessing the data, including text vectorization and numerical scaling.

## Usage

1. Enter the email content and metadata in the provided form on the web page.

2. Click on "Classify Email" to see whether the email is predicted as "Spam" or "Not Spam".

## Image

![Error: Image can't be loaded!](https://github.com/Ritaja-Jana/Email-Spam-Classification/blob/main/Image.png)



## Example of Spam Emails:


```bash
Subject: Urgent: Claim Your Cash Prize Now!

Dear Winner,

We are pleased to inform you that you've won a cash prize of $5,000! This is your chance to celebrate your good fortune. Follow the link provided to claim your winnings securely and instantly.

Congratulations again!

Warm regards,
Prize Rewards Team
```

```bash
Subject: Urgent Opportunity to Earn Quick Cash!

Dear Valued Customer,

You've been selected for an exclusive financial opportunity! Earn in just one week. Act now and secure your spot!

Best regards,
Financial Freedom Team
```

## 🔗 Connect On

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ritaja-jana)
