import time

import joblib
import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.layers import Layer

from model import attention_preprocess
from model.utils import export_training_plot, num_features, max_length, load_data, num_epochs

tf.random.set_seed(42)


# Custom Attention Layer class
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize weights for attention: W is a weight matrix, and b is a bias vector.
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        # Compute attention scores (energy score) using a tanh activation function
        # The tanh function helps to normalize the output between -1 and 1,
        # providing a balanced range which is beneficial for the next step in the attention process.
        energy_score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)

        # Removes an unnecessary singleton dimension from the attention scores, making it compatible with softmax
        energy_score = tf.squeeze(energy_score, axis=-1)

        # Convert scores to probabilities using softmax
        alpha = tf.nn.softmax(energy_score)

        # Apply the attention weights to the input data
        context = x * tf.expand_dims(alpha, -1)

        # Sum to get a weighted representation
        context = tf.reduce_sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        # Output shape after applying attention mechanism
        return input_shape[0], input_shape[-1]


def build_attention_model(vocab_size, max_length, label_size):
    # Model configuration parameters
    embedding_dim = 128

    # Build a sequential model with Keras
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_length,
            ),
            LSTM(
                256, return_sequences=True
            ),  # LSTM layer with return_sequences=True for attention
            Attention(),  # Custom attention layer
            Dense(
                512, activation="relu", kernel_regularizer="l2"
            ),  # Dense layer with L2 regularization
            Dropout(0.5),  # Dropout for regularization
            Dense(
                label_size, activation="softmax"
            ),  # Output layer with softmax activation
        ]
    )
    return model


def train_model():
    start_time = time.time()

    # Load and preprocess data
    documents, labels = load_data()
    X_train, X_test, y_train, y_test, tokenizer, one_hot_encoder = (
        attention_preprocess.preprocess_data(
            documents, labels, num_features=num_features, max_length=max_length
        )
    )

    # Vocabulary size adjustment
    vocab_size = len(tokenizer.word_index) + 1  # Account for reserved 0 index

    # Construct and compile the model
    model = build_attention_model(vocab_size, max_length, y_train.shape[1])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=64
    )

    # Plot and save training history
    model_name = "Attention Model with LSTM"
    export_training_plot(history, model_name)

    # Save model and serializers
    model.save(f"{model_name}.keras")
    joblib.dump(tokenizer, open("tokenizer.pkl", "wb"))
    joblib.dump(one_hot_encoder, open("one_hot_encoder.pkl", "wb"))

    # Print best training and validation accuracies
    print(f"{model_name}")
    print(f"Best training accuracy: {max(history.history['accuracy'])}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy'])}")

    end_time = time.time()
    # Print the time taken to train the model
    print(f"Time taken to train the model: {end_time - start_time} seconds")

    return tokenizer, one_hot_encoder


if __name__ == "__main__":
    train_model()
