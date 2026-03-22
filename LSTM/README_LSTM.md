# Model 1: Bidirectional LSTM for IMDB Sentiment Analysis

## The Model Architecture
This model utilizes a Recurrent Neural Network (RNN) architecture, specifically a Bidirectional LSTM, designed to capture the sequential context of movie reviews. The layers are structured as follows:
1. **Embedding Layer:** Utilizes pre-trained Google News Word2Vec vectors (300 dimensions). The weights are frozen to leverage the pre-learned semantic relationships of the vocabulary words without destroying them during training.
2. **Bidirectional LSTM Layer:** Contains 64 units. The bidirectional wrapper allows the network to process the text sequences in both forward and backward directions, capturing past and future context simultaneously.
3. **Dense Hidden Layer:** Contains 32 units with a ReLU activation function to learn non-linear representations from the LSTM output.
4. **Output Layer:** A single neuron with a Sigmoid activation function to output a binary probability (Positive/Negative sentiment).

## The Techniques Applied
To enhance performance and prevent overfitting, the following optimization techniques were applied:
* **Pretrained Word Embeddings:** Used Gensim to load the `word2vec-google-news-300` model, converting our text sequences into dense vectors. 
* **Optimization Algorithm:** The `Adam` optimizer was used with a tuned learning rate of `0.001` for efficient gradient descent.
* **Regularization (Dropout):** 50% Dropout layers (`Dropout(0.5)`) were added after both the LSTM layer and the Dense hidden layer. This randomly drops node weights during training, reducing the model's reliance on specific pathways and mitigating overfitting.
* **Early Stopping:** Training was monitored using the validation loss (`val_loss`). An Early Stopping callback was implemented with a `patience` of 3, meaning training automatically halted when the validation loss stopped improving for 3 consecutive epochs, and the best model weights were restored.
* **Hyperparameter Tuning:** Processed the data in batch sizes of 128 over a maximum of 15 epochs.

## Results
The model was evaluated on a held-out test set of 5,000 unseen reviews and achieved the following performance:
* **Test Accuracy:** **87.81%**
