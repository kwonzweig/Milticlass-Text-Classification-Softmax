import nltk
import numpy as np

from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Download necessary NLTK resources
nltk.download("reuters")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def load_data():
    """
    Loads and preprocesses the Reuters dataset. Cleans the text and extracts categories as labels.

    Returns:
        tuple: A tuple containing two lists - the cleaned documents and their corresponding labels.
    """
    documents = []
    labels = []

    # Iterate over all Reuters file IDs
    for file_id in reuters.fileids():
        documents.append(reuters.raw(file_id))
        labels.append(reuters.categories(file_id)[0])  # Taking the first category

    # Print the number of documents and unique labels
    print(f"Number of documents: {len(documents)}")
    print(f"Number of unique labels: {len(set(labels))}")

    return documents, labels


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        X (list): The list of documents.
        y (list): The list of labels.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing the split of X and y into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(documents, labels, num_features=1000):
    # Initialize and fit the one-hot encoder on labels
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(np.array(labels).reshape(-1, 1))

    # Split data into training and testing sets
    documents_train, documents_test, labels_train, labels_test = split_dataset(
        documents, labels
    )

    # Initialize and fit the tokenizer on the training documents only
    tfidf_vectorizer = TfidfVectorizer(max_features=num_features, stop_words="english")
    tfidf_vectorizer.fit(documents_train)

    # Pad sequences for both training and testing data
    X_train = tfidf_vectorizer.transform(documents_train).toarray()
    X_test = tfidf_vectorizer.transform(documents_test).toarray()

    # Convert labels using the fitted one-hot encoder
    y_train = one_hot_encoder.transform(np.array(labels_train).reshape(-1, 1))
    y_test = one_hot_encoder.transform(np.array(labels_test).reshape(-1, 1))

    # Print number of training and testing samples
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, tfidf_vectorizer, one_hot_encoder


def preprocess_prediction_data(text, tfidf_vectorizer):
    """
    Preprocesses the input text for making predictions.

    Args:
        text (str): The input text to be preprocessed.
        tfidf_vectorizer (TfidfVectorizer): The fitted TfidfVectorizer instance.

    Returns:
        np.ndarray: The padded sequence of the input text.
    """
    return tfidf_vectorizer.transform([text]).toarray()
