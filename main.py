import time

from model import attention_model
from model import tf_idf_model

start_time = time.time()
# Train the TF-IDF model without robust training
model = tf_idf_model.train_model(model_name="TF-IDF model", early_stopping=False)


# Train the TF-IDF model with robust training
model = tf_idf_model.train_model(
    model_name="TF-IDF with overfit prevention", early_stopping=False
)
end_time = time.time()
# Print the time taken to train the model
print(f"Time taken to train the model: {end_time - start_time} seconds")

start_time = time.time()
# Train the attention model
attention_model.train_model()
end_time = time.time()
# Print the time taken to train the model
print(f"Time taken to train the model: {end_time - start_time} seconds")
