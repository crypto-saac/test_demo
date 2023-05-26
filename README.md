Introduction

Recurrent Neural Networks (RNNs) are a powerful class of neural networks designed specifically to model sequential data. Their unique ability to incorporate previous inputs into the current output makes them unparalleled in their predictive capabilities for sequential data. In contrast, feedforward neural networks lack memory of previous inputs, rendering them ineffective in predicting future events.

An extension of RNNs is the Long Short-Term Memory (LSTM) network, which introduces memory cells with three distinct gates: input, forget, and output gates. These gates control the flow of information, allowing new information to enter through the input gate, deleting information that is irrelevant through the forget gate, and regulating the output of the current timestep through the output gate. The use of the sigmoid activation function in LSTM models enables efficient backpropagation, while simultaneously addressing the problem of vanishing gradients through gradient clipping.

Another type of RNN that addresses the vanishing gradient problem is the Gated Recurrent Unit (GRU). GRUs have advantages over LSTMs in terms of computational efficiency and memory usage, but LSTMs outperform GRUs on datasets with longer sequences. GRUs employ two gates, the update and reset gates, which allow them to selectively pass information from previous timesteps down the chain of events to make more accurate predictions.

Word embedding is a set of language modelling and feature learning techniques used to map words or phrases to real numbers. These word embeddings are used as inputs for LSTM and GRU models during training and testing, which further enhances their predictive capabilities.

Objective:

In this experiment we will explore the LSTM and GRU by comparing and contrasting which model is better than the other in the context of sentiment analysis as a sequence-to-one classification problem using various evaluation metrics.

Data preprocessing

The first step to implement a solution is to acquire data. In this case a dataset was readily available and provided for this experiment. The use of data frames to load and store this data proved to be efficient as it was easier to access this data upon request. To be explicit, this data was stored in a temporary structure defined as tmp_store which is a data frame. As part of preprocessing, we used a function defined as preprocess_text which takes in a text (“sequence”) and executed processes such as removing punctuations, digits, stop words, URLS and HTML tags and converting every character to lower case in the sequence. Snowball Stemmer was used to reduce the number of unique words that need to be processed. Reducing words to their base form, variations of the same words (such as “running”, “runs”, “ran”) can be treated as the same word, which can improve the accuracy and efficiency of the text analysis.

Tokenizing the text is used as building a vocabulary of words or subwords, encoding the text as sequences of integers and padding the sequences to a fixed length.

Pre-training

Before we train or build our model we had to download and extract the ‘glove.6B.zip’ file which enabled us to use the Glove model. The glove file that was downloaded and extracted is a pre-trained word embedding model that is used to learn word vectors that captures the semantic and relationships between words. A common usage of these vectors is sentiment analysis which is what is required in this experiment.

gl_em_dict is a function that was used to create a dictionary of word embeddings using pre-trained Glove vectors from a text file. This dictionary enables mapping of each word in the Glove pre-trained model to its corresponding vector representation.

em_matrix : This matrix contains vector representation of each word in the vocabulary filled with zeros if a word does not have a corresponding glove representation and assigned values if it does.

The embedded using embedding from tensorflow to convert em_matrix into weights that will be used by the LSTM and GRU models.

Training

LSTM: In building and training this model we used an embedding layer, two LSTM layers and a several fully connected Dense layers.

The architecture of the model is as follows:

The input layer receives a sequence of integer values.

The input sequence is fed into an embedding layer which performs various functions such as converting integer-coded inputs into dense vectors of fixed size where each integer corresponds to the word in the vocabulary.

The SpatialDropout1D layers applies dropout regularization to the embedding and prevents overfitting.

The two LSTMs both have 64 units each. With the first LSTM layer returning a full sequence of outputs for each time step and the second layer only focusing on returning the final output.

Learning is improved by the use of the two Dense layer and the ReLu activation.

In the output the use of the sigmoid activation is used to produce a binary classification output which was proven to be useful over the softmax activation function which is used to produce a probability.

GRU: In building and training this model we used an embedding layer and a Gated Recurrent Unit layer.

The architecture of the model is as follows:

The model is built using the Sequential API in Keras which allows for easy layer-by-layer construction of neural networks.

The first layer of this model is an embedding layer and this learns a dense representation of the input words in the form of word embeddings.

The GRU layer is then added on top of the embedding layer which processes the input sequence and learns to long term dependencies in the data.

The Dense layer in the output is a single output unit and the sigmoid activation function is used to produce a binary classification prediction.

Both the LSTM and GRU use the compile function to optimize and evaluation metric “accuracy” to monitor during training. The “fit” method takes the training data as input and updates the model parameters to minimize the loss on the training data.

Evaluation

To evaluate these models for appropriation comparison, the implementation of various evaluation metrics was deemed necessary. The evaluation metrices that are implemented are the F-1 score, precision_score, accuracy_score, recall_score and auc_roc.

F1-score and AUC-ROC are measures of the overall performance of a binary classification model, while precision_score and recall_score are measures of the model's performance on positive examples. These metrics are useful for evaluating the performance of a model and for comparing different models.

With the use of all these measures of performance we will then be to evaluate and conclude which is the optimal model without being baised over one evaluation metric’s findings.

Conclusion

These results were taken after running each model for two epochs. Since the GRU has evaluation metrices that are greater than the LSTMs we will conclude that for this Experiment the GRU proved to be the optimal model. In addition the GRU model trains faster than the LSTM model using the standard runtime in google colab.

