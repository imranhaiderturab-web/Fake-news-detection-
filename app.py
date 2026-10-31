import streamlit as st
import joblib 
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf

# Page Configuration

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# Load Model and Vectorizer

import os
import joblib

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model and vectorizer file paths
model_path = os.path.join(BASE_DIR, "nlp_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "nlp_vectorizer.pkl")

# Debugging information
print("App folder:", BASE_DIR)
print("Files in folder:", os.listdir(BASE_DIR))
print("Model path:", model_path)
print("Model exists:", os.path.isfile(model_path))
print("Vectorizer path:", vectorizer_path)
print("Vectorizer exists:", os.path.isfile(vectorizer_path))

# Load model
if os.path.isfile(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load vectorizer
if os.path.isfile(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    print("Vectorizer loaded successfully!")
else:
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")


# Custom CSS


st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="main-title">Fake News Detection System</div>',
    unsafe_allow_html=True
)

# Tabs
tab1 = st.tabs(
    ["Predict News"]
)[0]


# PREDICTION TAB

with tab1:

    st.header("Check News Article")

    Title = st.text_input(
        "News Title",
        placeholder="Enter the news headline..."
    )

    clean_text = st.text_area(
        "News Content",
        placeholder="Enter the complete news article here...",
        height=250
    )

    predict_button = st.button(
        "Predict News",
        use_container_width=True
    )

    if predict_button:

        if Title.strip() == "" and clean_text.strip() == "":
            st.warning("Please enter the news title or article text.")

        else:

            # Combine Title and Clean Text
            combined_text = Title + " " + clean_text

            # Transform using trained TF-IDF
            X_new = vectorizer.transform([combined_text])

            # Prediction
            prediction = model.predict(X_new)[0]

            # Probability
            probabilities = model.predict_proba(X_new)[0]

            
            # Handle labels
           

            if str(prediction).lower() in ["fake", "0"]:
                label = "FAKE NEWS"
                confidence = probabilities[list(model.classes_).index(prediction)]

            else:
                label = "REAL NEWS"
                confidence = probabilities[list(model.classes_).index(prediction)]

         
            # Result
       

            st.divider()

            if label == "FAKE NEWS":

                st.error(
                    f" Prediction: {label}"
                )

            else:

                st.success(
                    f" Prediction: {label}"
                )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            
            # Probability
        

            st.subheader("Prediction Probability")

            classes = model.classes_

            for cls, probability in zip(classes, probabilities):

                if str(cls).lower() in ["fake", "0"]:
                    class_name = "Fake"
                else:
                    class_name = "Real"

                st.write(
                    f"**{class_name}: {probability * 100:.2f}%**"
                )

                st.progress(float(probability))






