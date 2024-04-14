
# Reuters Text Classification

This project uses the Reuters Corpus dataset from the NLTK library to train a deep learning model for text classification. It utilizes TensorFlow to build and train a model that can classify text documents into various categories based on their content. 

## Reuters Corpus Dataset
Reuters Corpus is a collection of news documents with categories assigned to each document.
I took first label of each document as the target label, making it a suitable dataset for softmax output layer.

```
number of documents: 10788
number of categories: 90
test size for data split: 20%
```

## Model Development

### 1. Data Preprocessing

- **TF-IDF**: Converts text data into numerical vectors, capturing the importance of words in the document.
- **One-Hot Encoding**: Converts categorical labels to a binary matrix, necessary for categorical output in classification tasks.


**Improvement**: Try and compare the result with alternative NLP techniques such as Word2Vec and Bag of Words(BoW).

### 2. Model Training

- **ReLU with L2 regularization**: Activation function that introduces non-linearity and helps the model learn complex patterns. L2 regularization helps prevent overfitting by penalizing large weights.
- **Dropout Layer**: Regularizes the model by randomly setting a fraction of input units to zero during training.
- **BatchNormalization**: Normalizes the activations of the previous layer at each batch, improving the stability and speed of training.
- **Output Layer with Softmax Activation**: Produces probabilities across the output classes for classification.
- **Why L2 Reg over L1 Reg**: L2 regularization is more effective in preventing overfitting by penalizing large weights more smoothly, whereas L1 regularization tends to produce sparse weights (many zeros).

**Improvement**: Experiment with different architectures like LSTM, GRU, or Transformer models to capture the sequential nature of text data more effectively.


### 3. Model Evaluation

- **Loss Function**: Categorical Crossentropy, because it is suitable for multi-class classification problems where each class is mutually exclusive.
- **Optimizer**: Adam, as it automatically adjusts the learning rate and is well-suited for problems with large data and parameters.
- **Metrics**: Accuracy, a common metric for evaluating classification models, to provide a straightforward interpretation of model performance.

**Improvement**: Experiment with different evaluation metrics like precision, recall, and F1-score to understand the model's performance across different classes.

## Training Result

### TF-IDF model

![TF-IDF Model Accuracy Plot](https://raw.githubusercontent.com/kwonzweig/Milticlass-Text-Classification-Softmax/master/TF-IDF%20with%20overfit%20prevention_accuracy_plot.png)

```python
model = Sequential(
    [
        Dense(512, activation="relu", kernel_regularizer=l2(0.01)), # Dense layer with L2 regularization
        Dropout(0.5), # Dropout for regularization
        Dense(256, activation="relu", kernel_regularizer=l2(0.01)), # Dense layer with L2 regularization
        Dropout(0.5), # Dropout for regularization
        Dense(label_size, activation="softmax"), # Output layer with softmax activation
    ]
)
```

```
TF-IDF
Best training accuracy: 0.67
Best validation accuracy: 0.70
Time taken to train the model: 57.70 seconds
```

### Attention model with LSTM

![Attention Model with LSTM Accuracy Plot](https://raw.githubusercontent.com/kwonzweig/Milticlass-Text-Classification-Softmax/master/Attention%20Model%20with%20LSTM_accuracy_plot.png)


```python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
    LSTM(256, return_sequences=True),  # LSTM layer with return_sequences=True for attention
    Attention(),  # Custom attention layer
    Dense(512, activation="relu", kernel_regularizer="l2"),  # Dense layer with L2 regularization
    Dropout(0.5),  # Dropout for regularization
    Dense(label_size, activation="softmax"), # Output layer with softmax activation
])
```

```
Attention Model with LSTM
Best training accuracy: 0.93
Best validation accuracy: 0.84
Time taken to train the model: 740.75 seconds
```

### Improvement
- Try Cross Validation to get a better estimate of the model's performance by training and evaluating the model on different subsets of the data.
- Hyperparameter tuning to find the best combination of hyperparameters for the model.