# Keyword-Identification-using-SIAMESE-network
created an algorithm, that takes html page as input and infers if the page contains the information about cancer tumorboard or not

## MODEL
Here, we define our model, as a siamese network. The model is a sequence of layers, starting with a TextVectorization layer. This layer accepts natural language (text) as input, and maps it to an integer sequence. At initialization time, we should provide a vocabulary of words for it to be able to map the words at prediction time.

Following the text vectorization layer, we implement three Dense layers, with two Dropout layers in between. Lastly, we apply a L2 normalization layer to penalize large weights.

In our implementation of a siamese network, we override the call method of the tf.keras.Model class. This is needed because of the nature of the model.

Siamese networks take as input triplets: anchor (baseline) input, a sample from the same class as the anchor - positive, and a sample from a different class than the anchor - negative. It then does two passes the anchor twice through the network: once in combination with the positive sample, and the second time with the negative sample. Lastly, it compares the difference in outputs from the two passes. We expect the error/loss of the model to be low for the "positive pass" and higher for the "negative pass" since we want samples from the same class to be as similar to each other as possible, and as different from other classes as possible.
