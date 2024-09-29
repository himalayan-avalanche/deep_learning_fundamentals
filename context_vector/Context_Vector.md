# Whats context vector?

If you look at RNN or transformer based architecture, its often based on Encoder-Decoder archiecture (and lot in between). The Encoder and Decoder is often stack of RNN cells, 
or multi head self attention blocks used to generate the context vector. The context vector along with hidden state output concatenated and then passed through a linear layer to generate the vector same size as vocabulary.
This vocab size vector is then passed to a softmax function to generate the probabilities vector, which in turn is then used to find most likely word/token prediction.

As it turns out, the context vector is at core of any DL architecture used to make prediction. In fact for all the different DL architectures of Bidirectional RNNs, GRUs, LSTM, Transformers, the goal is to stack hidden layer, define transformations (such as using self attention to use all previous hidden layers than just last) to generate a better context vector. Of course choice of architecture also allows to scale the model traning such as use of  self attention mechanism (where attention scores are independent of tokens, thus removes the sequential dependence of forward calls and parameter updates).
