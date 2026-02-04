import os
import pandas as pd
from fastapi import UploadFile
from src.preprocessing.features import extract_features
from src.preprocessing.labeling import label_eeg_states
from src.preprocessing.load_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.core.ml.model import load_calibrated_model

def process_uploaded_file(uploaded_file: UploadFile):
    """Handles uploaded EEG file, processes it, and runs model inference."""
    file_location = f"temp/{uploaded_file.filename}"
    os.makedirs("temp", exist_ok=True)

    # Save file to temp directory
    with open(file_location, "wb") as f:
        f.write(uploaded_file.file.read())

    # Load EEG data
    df = load_data(file_location)
    df = label_eeg_states(df)
    print(df.describe())  # Fixed typo

    # Extract features
    features_df = extract_features(df)
    print(features_df)

    # Preprocess data
    X, _, _, _, metadata = preprocess_data(features_df)

    # Load trained model
    model_path = "./data/processed/trained_model.h5"
    model = load_calibrated_model(model_path)

    # Predict mental state
    predictions = model.predict(X.reshape(-1, X.shape[1], 1))
    predicted_state = predictions.argmax(axis=1)

    # Clean up temporary file
    os.remove(file_location)

    return {"predicted_state": predicted_state.tolist()}
