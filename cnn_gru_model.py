'''
Created on June 20, 2017

@author: Hao Wu
'''
from keras.layers import Input, TimeDistributed
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.core import Reshape, Flatten, Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import Average, Concatenate
from keras.layers.recurrent import GRU
from keras.models import Model
import numpy as np


# from keras.preprocessing.sequence import pad_sequences

class CNN_GRU_module():
    epochs = 1
    batch_size = 128
    def __init__(self,
                 output_dimesion,
                 vocab_size,
                 dropout_rate,
                 emb_dim,
                 gru_outdim,
                 maxlen_doc,
                 maxlen_sent,
                 nb_filters,
                 init_W=None):
        self.filter_lengths = [3, 4, 5]
        if init_W is not None:
            self.vocab_size, self.emb_dim = init_W.shape
        else:
            self.vocab_size = vocab_size
            self.emb_dim = emb_dim
        self.maxlen_doc = maxlen_doc
        self.maxlen_sent = maxlen_sent
        self.nb_filters = nb_filters
        self.gru_outdim = gru_outdim

        print ("Build model...")
        """Embedding Layers"""
        in_seq = Input(shape=(self.maxlen_doc * self.maxlen_sent,))
        if init_W is None:
            seq_emb = Embedding(self.vocab_size, self.emb_dim, trainable=True)(in_seq)
        else:
            seq_emb = Embedding(self.vocab_size, self.emb_dim, weights=[init_W / 20], trainable=False)(in_seq)
        seq_emb = Reshape((self.maxlen_doc, self.maxlen_sent, self.emb_dim, 1))(seq_emb)
        """CNN Layers"""
        tmp_list = []
        for ws in self.filter_lengths:
            cnn = TimeDistributed(Convolution2D(self.nb_filters, ws, emb_dim, border_mode='valid', activation='relu'))(seq_emb)
            cnn = TimeDistributed(MaxPooling2D(pool_size=(self.maxlen_sent - ws + 1, 1), border_mode='valid'))(cnn)
            cnn = TimeDistributed(Flatten())(cnn)
            cnn = TimeDistributed(Activation('tanh'))(cnn)
            tmp_list.append(cnn)
        cnn_con = Concatenate()(tmp_list)
        """GRNN Layers"""
        h_forward = GRU(self.gru_outdim) (cnn_con)
        h_backward = GRU(self.gru_outdim, go_backwards=True)(cnn_con)
        h = Concatenate()([h_forward, h_backward])
        """Output Layer"""
        seq_dropout = Dropout(dropout_rate)(h)
        out_seq = Dense(output_dimesion, activation='tanh')(seq_dropout)
        # build and compile model
        self.model = Model(in_seq, out_seq)
        self.model.compile(optimizer='rmsprop', loss='mse')
        print(self.model.summary())
        
    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def train(self, X_train, V, weight_of_sample, seed):
        np.random.seed(seed) 
        X_train = np.random.permutation(X_train)
        np.random.seed(seed) 
        V = np.random.permutation(V) 
        np.random.seed(seed)
        weight_of_sample = np.random.permutation(weight_of_sample)

        print("Train CNN_GRU_module...")
        history = self.model.fit(X_train, V,
                                 verbose=1, batch_size=self.batch_size, epochs=self.epochs, sample_weight=weight_of_sample)
        return history


    def get_projection_layer(self, X_train):
        theta = self.model.predict(X_train, batch_size=self.batch_size)
        return theta
    

