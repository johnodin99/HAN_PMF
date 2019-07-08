#coding=utf8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

def length(sequences):
#返回一个序列中每个元素的长度
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, vocab_size, num_classes, embedding_size=30, hidden_size=10, seed=None):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seed = seed
        self.input_word = None
        self.word_alpha = None

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            #x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            #y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')

        #构建模型
        word_embedded, self.input_word = self.word2vec()
        sent_vec, self.word_alpha = self.sent2vec(word_embedded)
        doc_vec, sentence_alpha = self.doc2vec(sent_vec)
        out = self.classifer(doc_vec)
        self.out = out

    def word2vec(self):
        #嵌入层
        with tf.name_scope("embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal(shape=(self.vocab_size, self.embedding_size),
                                                            mean=0.0,
                                                            stddev=1.0,
                                                            dtype=tf.float32,
                                                            seed=self.seed,
                                                            name=None))

            #shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
        return word_embedded, self.input_x

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            #GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            #batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            #并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            #shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(tensor=word_embedded,
                                       shape=[-1, self.max_sentence_length, self.embedding_size])
            #shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalGRUEncoder(inputs=word_embedded,
                                                        name='word_encoder')
            #shape为[batch_size*sent_in_doc, hidden_size*2]
            sent_vec, alpha = self.AttentionLayer(inputs=word_encoded,
                                           name='word_attention')
            return sent_vec, alpha

    def doc2vec(self, sent_vec):
        #原理与sent2vec一样，根据文档中所有句子的向量构成一个文档向量
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            #shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            #shape为[batch_szie, hidden_szie*2]
            doc_vec, alpha = self.AttentionLayer(doc_encoded, name='sent_attention')
            return doc_vec, alpha

    def classifer(self, doc_vec):
        #最终的输出层，是一个全连接层
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec,
                                         num_outputs=self.num_classes,
                                         activation_fn=None)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        #双向GRU的编码层，将一句话中的所有单词或者一个文档中的所有句子向量进行编码得到一个 2×hidden_size的输出向量，然后在经过Attention层，
        # 将所有的单词或句子的输出向量加权得到一个最终的句子/文档向量。
        #输入inputs的shape是[batch_size, max_time, voc_size]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            GRU_cell_fw = rnn.GRUCell(num_units=self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(num_units=self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), axis=2)
            return outputs

    def AttentionLayer(self, inputs, name):
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2], seed=self.seed), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs=inputs,
                                       num_outputs=self.hidden_size * 2,
                                       activation_fn=tf.nn.tanh)

            # shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output, alpha




