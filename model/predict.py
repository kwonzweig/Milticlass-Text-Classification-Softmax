import json

import joblib
import numpy as np
from keras.src.saving import load_model

from model.tf_idf_preprocess import preprocess_prediction_data


def load_trained_model(model_path="reuters_model.keras"):
    model = load_model(model_path)
    # Read tokenizer from 'tokenizer.json'
    tokenizer = json.loads(open("tokenizer.json").read())
    # Read 'encoder.pkl'
    encoder = joblib.load("one_hot_encoder.pkl")
    return tokenizer, encoder, model


def make_prediction(text, tokenizer, encoder, model):
    # Preprocess the input text
    padded_seq = preprocess_prediction_data(text, tokenizer)
    prediction = model.predict(padded_seq)
    label_idx = np.argmax(prediction)
    return encoder.inverse_transform([label_idx])[0]
