
# Reuters Text Classification

This project uses the Reuters Corpus dataset from the NLTK library to train a deep learning model for text classification. It utilizes TensorFlow to build and train a model that can classify text documents into various categories based on their content. The project is structured into four main scripts: `preprocess.py`, `training.py`, `predict.py`, and `streamlit_main.py`, each handling different aspects of the machine learning pipeline.

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


**Improvement**: Experiment with different architectures like LSTM, GRU, or Transformer models to capture the sequential nature of text data more effectively.


### 3. Model Evaluation

- **Loss Function**: Categorical Crossentropy, because it is suitable for multi-class classification problems where each class is mutually exclusive.
- **Optimizer**: Adam, as it automatically adjusts the learning rate and is well-suited for problems with large data and parameters.
- **Metrics**: Accuracy, a common metric for evaluating classification models, to provide a straightforward interpretation of model performance.

**Improvement**: Experiment with different evaluation metrics like precision, recall, and F1-score to understand the model's performance across different classes.

## Training Result

### TF-IDF model
```

```