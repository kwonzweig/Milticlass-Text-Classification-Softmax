from matplotlib import pyplot as plt
from nltk.corpus import reuters

max_length = 200
num_features = 4000
num_epochs = 20


def export_training_plot(history, name):
    # Plotting training and validation accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Model Accuracy ({name})")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.xticks(range(len(history.history["accuracy"])))
    plt.savefig(f"{name}_accuracy_plot.png")

    # Plotting training and validation loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Model Loss ({name})")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.xticks(range(len(history.history["loss"])))
    plt.savefig(f"{name}_training_plot.png")


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

    # Export as csv file
    import pandas as pd

    df = pd.DataFrame({"documents": documents, "labels": labels})
    df.to_csv("reuters.csv", index=False)

    return documents, labels
