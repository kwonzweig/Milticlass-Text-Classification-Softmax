import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.src.legacy.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Download necessary NLTK resources
nltk.download("reuters")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Set up stopwords and the lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Tokenize text
    words = word_tokenize(text)
    # Convert to lower case and remove punctuation
    words = [word.lower() for word in words if word.isalnum()]
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


def preprocess_data(documents, labels, num_features=1000, max_length=200):
    # Clean the documents
    documents = [clean_text(doc) for doc in documents]

    # Initialize and fit the one-hot encoder on labels
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(np.array(labels).reshape(-1, 1))

    # Split data into training and testing sets
    documents_train, documents_test, labels_train, labels_test = train_test_split(
        documents, labels, test_size=0.2, random_state=42
    )

    # Initialize and fit the tokenizer on the training documents only
    tokenizer = Tokenizer(num_words=num_features)
    tokenizer.fit_on_texts(documents_train)

    # Convert text documents to padded sequences
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(documents_train), maxlen=max_length
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(documents_test), maxlen=max_length
    )

    # Convert labels using the fitted one-hot encoder
    y_train = one_hot_encoder.transform(np.array(labels_train).reshape(-1, 1))
    y_test = one_hot_encoder.transform(np.array(labels_test).reshape(-1, 1))

    return X_train, X_test, y_train, y_test, tokenizer, one_hot_encoder


def preprocess_prediction_data(text, tokenizer, max_length=500):
    # Convert a single text input to a padded sequence
    return pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_length)
