import streamlit as st
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter

# Load client models
client_models = [joblib.load(f"client_model_{i}.pkl") for i in range(5)]

# Load the vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function for aggregating predictions using majority voting
def aggregate_predictions(models, X_input):
    predictions = np.array([model.predict(X_input) for model in models], dtype=object)
    final_predictions = []
    for i in range(predictions.shape[1]):
        sample_preds = predictions[:, i]
        most_common = Counter(sample_preds).most_common(1)[0][0]
        final_predictions.append(most_common)
    return np.array(final_predictions)

# Streamlit Interface
st.title("Phishing Email Detector")
st.write("Paste an email below to check if it's a Safe Email or a Phishing Email.")

# User input
user_input = st.text_area("Enter the email text:", "")

# Prediction
if st.button("Predict"):
    if user_input:
        # Transform input using the TF-IDF vectorizer
        input_tfidf = vectorizer.transform([user_input])

        # Get aggregated predictions
        final_prediction = aggregate_predictions(client_models, input_tfidf)

        # Display the result
        st.write(f"**Prediction:** {final_prediction[0]}")
    else:
        st.warning("Please enter some email text to make a prediction.")
