# Model 2: 1D Convolutional Neural Network (CNN) for IMDB Sentiment Analysis

## The Model Architecture
This model treats text similarly to a 1D image, using convolutional filters to slide across word sequences and detect localized features (like n-grams or specific sentiment-heavy phrases). The layers are structured as follows:
1. **Embedding Layer:** Utilizes pre-trained Google News Word2Vec vectors (300 dimensions). The weights are frozen to preserve the pre-trained spatial and semantic relationships between words.
2. **Conv1D Layer:** Uses 128 filters with a kernel size of 5 and a ReLU activation. This layer scans the embedded word sequences 5 words at a time to extract local sentiment patterns.
3. **GlobalMaxPooling1D Layer:** Down-samples the feature maps by taking the maximum value over the time dimension, effectively selecting the most prominent features detected by the convolutional layer.
4. **Dense Hidden Layer:** Contains 32 units with a ReLU activation function.
5. **Output Layer:** A single neuron with a Sigmoid activation function to classify the review as Positive or Negative.

## The Techniques Applied
To enhance performance and prevent overfitting, the following optimization techniques were applied:
* **Pretrained Word Embeddings:** Integrated the `word2vec-google-news-300` model via Gensim to map vocabulary to dense, semantically meaningful vectors.
* **Optimization Algorithm:** The `Adam` optimizer was utilized with a learning rate of `0.001`.
* **Hyperparameter Tuning:** Tuned the Convolutional layer to use a specific window size (`kernel_size=5`) to look at 5-word phrases at a time. The model processes sequences in batch sizes of 128. 
* **Regularization (Dropout):** A 50% Dropout layer (`Dropout(0.5)`) was applied before the final output layer to force the model to distribute its learning across multiple features, preventing it from memorizing the training data.
* **Early Stopping:** An Early Stopping callback monitored the `val_loss` with a `patience` of 3 to halt training when the model began to overfit, successfully restoring the weights from the optimal epoch.

## Results
The model was evaluated on a held-out test set of 5,000 unseen reviews and achieved the following performance:
* **Test Accuracy:** **87.58%**
