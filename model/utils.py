from matplotlib import pyplot as plt

max_length = 200
num_features = 5000


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
