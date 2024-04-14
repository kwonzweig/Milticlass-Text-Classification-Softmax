import time

import joblib
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

from model import tf_idf_preprocess
from model.utils import export_training_plot, num_features, load_data, num_epochs

tf.random.set_seed(42)


def build_model(label_size):
    # Build the model
    model = Sequential(
        [
            Dense(512, activation="relu"),
            Dense(256, activation="relu"),
            Dense(label_size, activation="softmax"),
        ]
    )

    return model


def build_model_robust(label_size):
    # Build the model
    model = Sequential(
        [
            Dense(
                512, activation="relu", kernel_regularizer=l2(0.01)
            ),  # L2 regularization
            Dropout(0.5),  # Dropout layer for regularization
            Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(
                label_size, activation="softmax"
            ),  # Output layer with softmax for multiclass classification
        ]
    )

    return model


def train_model(model_name, early_stopping=False):
    start_time = time.time()
    documents, labels = load_data()
    X_train, X_test, y_train, y_test, tfidf_vectorizer, one_hot_encoder = (
        tf_idf_preprocess.preprocess_data(documents, labels, num_features=num_features)
    )

    # Build the model
    label_size = y_train.shape[1]

    if model_name == "TF-IDF with overfit prevention":
        model = build_model_robust(label_size)
    else:
        model = build_model(label_size)

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    if early_stopping:
        # Setting up early stopping
        early_stopping_monitor = EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=num_epochs,
            batch_size=64,
            callbacks=[early_stopping_monitor],
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=num_epochs,
            batch_size=64,
        )

    export_training_plot(history, model_name)

    model.save(f"{model_name}.keras")
    joblib.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
    joblib.dump(one_hot_encoder, open("one_hot_encoder.pkl", "wb"))

    # Print the best training / validation accuracy
    print(f"{model_name}")

    # Find the result with the best validation accuracy
    best_val_acc = max(history.history["val_accuracy"])

    # Print the training and validation accuracy from the best result
    print(
        f"Best training accuracy: {history.history['accuracy'][history.history['val_accuracy'].index(best_val_acc)]}"
    )
    print(f"Best validation accuracy: {best_val_acc}")

    end_time = time.time()
    # Print the time taken to train the model
    print(f"Time taken to train the model: {end_time - start_time} seconds")

    return model


if __name__ == "__main__":
    train_model("TF-IDF with overfit prevention")
