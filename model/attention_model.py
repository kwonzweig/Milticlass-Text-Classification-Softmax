import joblib
import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.layers import Layer

from model import attention_preprocess
from model.utils import export_training_plot, num_features, max_length


# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize weights
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
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Apply a linear transformation and tanh activation
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        e = tf.squeeze(e, axis=-1)
        # Compute the softmax
        alpha = tf.nn.softmax(e)
        # Weight the input by the attention scores
        context = x * tf.expand_dims(alpha, -1)
        context = tf.reduce_sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def build_attention_model(vocab_size, embedding_dim, input_length, label_size):
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=input_length,
            ),
            LSTM(128, return_sequences=True),
            Attention(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(label_size, activation="softmax"),
        ]
    )
    return model


def train_model():
    documents, labels = attention_preprocess.load_data()

    X_train, X_test, y_train, y_test, tokenizer, one_hot_encoder = attention_preprocess.preprocess_data(
        documents, labels, num_features=num_features, max_length=max_length
    )

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    embedding_dim = 50
    model = build_attention_model(
        vocab_size, embedding_dim, max_length, y_train.shape[1]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64
    )
    name = "Attention Model with LSTM"
    export_training_plot(history, name)

    model.save("reuters_model_attention.keras")
    joblib.dump(tokenizer, open("tokenizer.pkl", "wb"))
    joblib.dump(one_hot_encoder, open("one_hot_encoder.pkl", "wb"))

    # Print the best training / validation accuracy
    print(f"{name}")
    print(f"Best training accuracy: {max(history.history['accuracy'])}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy'])}")

    return tokenizer, one_hot_encoder


if __name__ == "__main__":
    train_model()
