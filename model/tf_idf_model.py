import joblib
from keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

import tf_idf_preprocess
from model.utils import export_training_plot, num_features


def build_model(label_size):
    # Build the model
    model = Sequential(
        [
            Dense(512, activation="relu"),
            Dense(256, activation="relu"),
            Dense(
                label_size, activation="softmax"
            ),  # Output layer with softmax for multiclass classification
        ]
    )

    return model


def build_model_robust(label_size):
    # Build the model
    model = Sequential(
        [
            Dense(512, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            BatchNormalization(),
            Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            BatchNormalization(),
            Dense(
                label_size, activation="softmax"
            ),  # Output layer with softmax for multiclass classification
        ]
    )

    return model


def train_model():
    documents, labels = tf_idf_preprocess.load_data()
    X_train, X_test, y_train, y_test, tfidf_vectorizer, one_hot_encoder = tf_idf_preprocess.preprocess_data(
        documents, labels, num_features=num_features
    )

    # Build the model
    label_size = y_train.shape[1]

    name = "TF-IDF model with overfit prevention"

    if name == "TF-IDF with overfit prevention":
        model = build_model_robust(label_size)
    else:
        model = build_model(label_size)

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # # Setting up early stopping
    # early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min',
    #                                        restore_best_weights=True)
    # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
    #                     callbacks=[early_stopping_monitor])

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64
    )

    export_training_plot(history, name)

    model.save("reuters_model.keras")
    joblib.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
    joblib.dump(one_hot_encoder, open("one_hot_encoder.pkl", "wb"))

    # Print the best training / validation accuracy
    print(f"{name}")
    print(f"Best training accuracy: {max(history.history['accuracy'])}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy'])}")

    return tfidf_vectorizer, one_hot_encoder


if __name__ == "__main__":
    train_model()
