# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import *
from beam_search_decoder import BeamSearchDecoder
from helper import TrainingHelper
import numpy as py

class Neuralmodel:
    def __init__(self,extract_sentence_flag,vocab_size, batch_size, embed_size,learning_rate,decay_step,decay_rate,
                 max_num_sequence,sequence_length,filter_sizes,feature_map,hidden_size,document_length,beam_width,
                 attention_size,input_y2_max_length,clip_gradients=5.0,initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter:"""
        self.initializer = initializer

        """Basic"""
        self.extract_sentence_flag = extract_sentence_flag
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size

        """learning_rate"""
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        self.decay_step = decay_step
        self.decay_rate = decay_rate

        """Overfit"""
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.clip_gradients = clip_gradients

        """CNN (word)"""
        self.max_num_sequence = max_num_sequence
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.feature_map = feature_map

        """LSTM (sentence)"""
        self.hidden_size = hidden_size
        self.document_length = document_length

        """LSTM + MLP (labeling)"""

        """LSTM + Attention (generating)"""
        self.beam_width = beam_width
        self.attention_size = attention_size
        self.input_y2_max_length = input_y2_max_length

        """Input"""
        self.input_x = tf.placeholder(tf.int32, [None, self.max_num_sequence, self.sequence_length], name="input_x")

        if extract_sentence_flag:
            self.input_y1 = tf.placeholder(tf.int32, [None, self.max_num_sequence], name="input_y_sentence")
            self.input_y1_length = tf.placeholder(tf.int32, [None], name="input_y_length")
            self.mask = tf.sequence_mask(self.input_y1_length, self.max_num_sequence, dtype=tf.float32, name='masks')
        else:
            self.input_y2_length = tf.placeholder(tf.int32, [None], name="input_y_word_length")
            self.input_y2 = tf.placeholder(tf.int32, [None, self.input_y2_max_length], name="input_y_word")
            self.input_decoder_x = tf.placeholder(tf.int32, [None, self.input_y2_max_length], name="input_decoder_x")
            self.value_decoder_x = tf.placeholder(tf.int32, [None, self.document_length], name="value_decoder_x")
            self.mask = tf.sequence_mask(self.input_y2_length, self.input_y2_max_length, dtype=tf.float32, name='masks')

        """Count"""
        self.global_step = tf.Variable(0, trainable=False, name='Global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.global_increment = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1)))

        """Process"""
        self.instantiate_weights()

        """Logits"""
        if extract_sentence_flag:
            self.logits = self.inference()
        else:
            self.logits, self.final_sequence_lengths = self.inference()

        if extract_sentence_flag:
            print('using sentence extractor...')
            self.loss_val = self.loss_sentence()
        else:
            print('using word extractor...')
            self.loss_val = self.loss_word()

        self.train_op = self.train()

    def instantiate_weights(self):
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer)
            self.Embedding_ = tf.get_variable("Embedding_", shape=[2, self.hidden_size], initializer=self.initializer)

        with tf.name_scope("lstm_cell"):
            # input gate
            self.W_i = tf.get_variable("W_i", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.U_i = tf.get_variable("U_i", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.b_i = tf.get_variable("b_i", shape=[self.hidden_size])
            # forget gate
            self.W_f = tf.get_variable("W_f", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.U_f = tf.get_variable("U_f", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.b_f = tf.get_variable("b_f", shape=[self.hidden_size])
            # cell gate
            self.W_c = tf.get_variable("W_c", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.U_c = tf.get_variable("U_c", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.b_c = tf.get_variable("b_c", shape=[self.hidden_size])
            # output gate
            self.W_o = tf.get_variable("W_o", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.U_o = tf.get_variable("U_o", shape=[self.hidden_size,self.hidden_size], initializer=self.initializer)
            self.b_o = tf.get_variable("b_o", shape=[self.hidden_size])

    def document_reader(self):
        """1.embedding"""
        # self.input_x : [batch_size, max_num_sequence, sentence_length]
        # self.embedded_words : [max_num_sequence, sentence_length, embed_size]
        # self.embedded_words_expanded : [batch_size, max_num_sequence, sentence_length, embed_size]
        embedded_words = []
        for idx in range(self.batch_size):
            self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x[idx:idx+1])
            self.embedded_words_squeezed = tf.squeeze(self.embedded_words, axis=0)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words_squeezed, axis=-1)
            embedded_words.append(self.embedded_words_expanded)

        #tf.scalar_mul(self.embedded_words_expanded,self.embedded_words)
        """2.CNN(word)"""
        # conv: [max_num_sequence, sequence_length-filter_size+1, 1, num_filters]
        # pooled: [max_num_sequence, 1, 1, num_filters]
        # pooled_temp: [max_num_sequence, num_filters * class_filters]
        # cnn_outputs: [batch_size, max_num_sequence, num_filters * class_filters]
        pooled_outputs = []
        for conv_s in embedded_words:
            pooled_temp = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("convolution-pooling-%s" % filter_size, reuse=tf.AUTO_REUSE):
                    filter=tf.get_variable("filter-%s"%filter_size,[filter_size,self.embed_size,1,self.feature_map[i]],initializer=self.initializer)
                    conv=tf.nn.conv2d(conv_s, filter, strides=[1,1,1,1], padding="VALID",name="conv")
                    conv=tf.contrib.layers.batch_norm(conv, is_training = self.is_training, scope='cnn_bn_')
                    b=tf.get_variable("b-%s"%filter_size,[self.feature_map[i]])
                    h=tf.nn.tanh(tf.nn.bias_add(conv,b),"tanh")
                    pooled=tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")
                    pooled_temp.append(pooled)
            pooled_temp = tf.concat(pooled_temp, axis=3)
            pooled_temp = tf.reshape(pooled_temp, [-1, self.hidden_size])
            pooled_outputs.append(pooled_temp)
        cnn_outputs = tf.stack(pooled_outputs, axis=0)

        """3.LSTM(sentence)"""
        # lstm_outputs: [batch_size, max_time, hidden_size]
        # cell_state: [batch_size, hidden_size]
        with tf.name_scope("lstm-encoder"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = self.dropout_keep_prob)
            lstm_outputs, cell_state = tf.nn.dynamic_rnn(lstm_cell, cnn_outputs, dtype = tf.float32)
        return cnn_outputs, lstm_outputs, cell_state

    def lstm_single_step(self, St, At, h_t_minus_1, c_t_minus_1, p_t_minus_1):

        # Xt = p_t_minus_1 * St
        p_t_minus_1 = tf.reshape(p_t_minus_1, [-1, 1])
        Xt = tf.multiply(p_t_minus_1, St)
        # dropout
        Xt = tf.nn.dropout(Xt, self.dropout_keep_prob)
        # input forget output compute
        i_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_i) + tf.matmul(h_t_minus_1, self.U_i) + self.b_i)
        f_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_f) + tf.matmul(h_t_minus_1, self.U_f) + self.b_f)
        c_t_candidate = tf.nn.tanh(tf.matmul(Xt, self.W_c) + tf.matmul(h_t_minus_1, self.U_c) + self.b_c)
        c_t = f_t * c_t_minus_1 + i_t * c_t_candidate
        o_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_o) + tf.matmul(h_t_minus_1, self.U_o) + self.b_o)
        h_t = o_t * tf.nn.tanh(c_t)
        # prob compute
        concat_h = tf.concat([At, h_t], axis=1)
        concat_h_dropout = tf.nn.dropout(concat_h, keep_prob=self.dropout_keep_prob)
        p_t = tf.layers.dense(concat_h_dropout, 1, activation=tf.sigmoid)

        return h_t, c_t, p_t

    def sentence_extractor(self):
        """4.1.1 LSTM(decoder)"""
        # decoder input each time: activation (MLP(h_t:At)) * St
        # h_t: decoder LSTM output
        # At: encoder LSTM output (document level)
        # St: encoder CNN output (sentence level)
        # probability value: [p_t = activation(MLP(h_t:At)) for h_t in h_t_steps ]

        # initialize
        h_t_lstm_list = []
        lstm_tuple = self.initial_state
        c_t = lstm_tuple[0]
        h_t = lstm_tuple[1]
        p_t = tf.ones((self.batch_size))
        cnn_outputs = tf.split(self.cnn_outputs, self.max_num_sequence, axis=1)
        cnn_outputs = [tf.squeeze(i, axis=1) for i in cnn_outputs]
        attention_state = tf.split(self.attention_state, self.max_num_sequence, axis=1)
        attention_state = [tf.squeeze(i, axis=1) for i in attention_state]
        # first step
        start_tokens = tf.zeros([self.batch_size], tf.int32) # id for ['GO']
        St = tf.nn.embedding_lookup(self.Embedding_, start_tokens)
        #tf.scalar_mul(St, St)
        At = attention_state[0]
        h_t, c_t, p_t = self.lstm_single_step(St, At, h_t, c_t, p_t)
        h_t_lstm_list.append(h_t)
        # next steps
        for time_step, merge in enumerate(zip(cnn_outputs[:-1], attention_state[1:])):
            St, At = merge[0], merge[1]
            h_t, c_t, p_t = self.lstm_single_step(St, At, h_t, c_t, p_t)
            h_t_lstm_list.append(h_t)
        # results
        decoder_outputs = tf.stack(h_t_lstm_list, axis=1)

        """4.1.2 MLP(score)"""
        # concat_outputs1: [batch_size, max_num_sequence, hidden_size*2]
        # concat_outputs2: [batch_size, max_num_sequence*(hidden_size*2)]
        # logits: [batch_size, max_num_sequence]
        with tf.name_scope("mlp-sentence-decoder"):
            concat_outputs1 = tf.concat([decoder_outputs, self.attention_state], axis = 2)
            concat_outputs2 = tf.reshape(concat_outputs1, [-1, self.max_num_sequence * self.hidden_size * 2])
            concat_outputs3 = tf.nn.dropout(concat_outputs2, keep_prob=self.dropout_keep_prob)
            logits = tf.layers.dense(concat_outputs3, self.max_num_sequence, activation=tf.sigmoid, use_bias=True)
        return logits

    def word_extractor(self):
        # LSTM inputs: h_t = LSTM(wt-1,h_t-1)
        # Attention: h~t = Attention(h_t,h)
        with tf.name_scope("attention-word-decoder"):
            """4.2 beam search preparation"""
            attention_state = tf.contrib.seq2seq.tile_batch(self.attention_state, multiplier=self.beam_width)
            encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.max_num_sequence, multiplier=self.beam_width)
            encoder_final_state = nest.map_structure(lambda s: seq2seq.tile_batch(s, self.beam_width), self.initial_state)
            """4.2 Attention(Bahdanau)"""
            # building attention cell
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            attention_mechanism1 = attention_wrapper.BahdanauAttention(
            num_units=self.hidden_size, memory=attention_state, memory_sequence_length=encoder_inputs_length
            )
            attention_cell = attention_wrapper.AttentionWrapper(
            cell=lstm_cell, attention_mechanism=attention_mechanism1, attention_layer_size=self.attention_size,                             \
            num_units=self.hidden_size, memory=attention_state, memory_sequence_length=encoder_inputs_length
            )
            attention_cell = attention_wrapper.AttentionWrapper(
            cell=lstm_cell, attention_mechanism=attention_mechanism1, attention_layer_size=self.attention_size,                             \
            # cell_input_fn=(lambda inputs, attention: tf.layers.Dense(self.hidden_size, dtype=tf.float32, name="attention_inputs")(array.ops.concat([inputs, attention],-1))) TODO \
            cell_input_fn=(lambda inputs, attention: tf.layers.Dense(self.hidden_size, dtype=tf.float32, name="attention_inputs")(inputs)), \
            alignment_history=False, name='Attention_Wrapper'                                                                               \
            )

            decoder_initial_state = attention_cell.zero_state(batch_size=(self.batch_size * self.beam_width), dtype=tf.float32).clone(cell_state=encoder_final_state)
            # inputs_decoder_embedded: [batch_size, input_y2_max_length]
            # values_decoder_embedded: [batch_size, document_length]
            inputs_decoder_embedded = tf.nn.embedding_lookup(self.Embedding, self.input_decoder_x)
            values_decoder_embedded = tf.nn.embedding_lookup(self.Embedding, self.value_decoder_x)

            def training_decode():
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=inputs_decoder_embedded, sequence_length=self.decoder_length, time_major=False, name="training_helper")
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell,helper=helper,initial_state=decoder_initial_state,output_layer=None)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,output_time_major=False,impute_finished=True,maximum_iterations=self.input_y2_max_length)
                return decoder_outputs
            def beamsearch_decode():
                start_tokens=tf.ones([self.batch_size,], tf.int32) * 2 #self.word_to_idx['START']
                end_token= 3 #self.word_to_idx['STOP']
                inference_decoder = BeamSearchDecoder(cell=attention_cell,embedding=values_decoder_embedded,start_tokens=start_tokens,end_token=end_token,initial_state=decoder_initial_state,beam_width=self.beam_width,output_layer=None)
                decoder_outputs, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,output_time_major=False,impute_finished=True,maximum_iterations=self.input_y2_max_length)
                return decoder_outputs
            decoder_outputs = tf.cond(self.is_training, training_decode, beamsearch_decode)

        """4.2 attention * document mat"""
        # decoder_outputs: [batch_size, input_y2_max_length, attention_size]
        # final_sequence_lengths: [batch_size]
        # logits: [batch_size, input_y2_max_length, document_length]
        with tf.name_scope("attention-vocab"):
            attention_mechanism2 =attention_wrapper.BahdanauAttention(
            num_units=self.attention_size, memory=values_decoder_embedded, memory_sequence_length=self.document_length
            )
            logits = attention_mechanism2(decoder_outputs)

        return logits, final_sequence_lengths

    def inference(self):
        """
        compute graph:
        1.Embedding--> 2.CNN(word)-->3.LSTM(sentence) (Document Reader)
        4.1 LSTM + MLP(labeling)                      (Sentence Extractor)
        4.2 LSTM + Attention(generating)              (Word Extractor)
        """
        self.cnn_outputs, self.attention_state, self.initial_state = self.document_reader()
        if self.extract_sentence_flag:
            logits = self.sentence_extractor()
            return logits
        else:
            logits, final_sequence_lengths = self.word_extractor()
            return logits, final_sequence_lengths

    def loss_sentence(self, label_smoothing=0,l2_lambda=0.001):
        # multi_class_labels: [batch_size, max_num_sequence]
        # logits: [batch_size, max_num_sequence]
        # losses: [batch_size, max_num_sequence]
        # sigmoid log: max(x, 0) + x * z + log(1 + exp(-x))
        with tf.name_scope("loss_sentence"):
            logits = tf.convert_to_tensor(self.logits)
            labels = tf.cast(self.input_y1, logits.dtype)
            if label_smoothing > 0:
                labels = (labels * (1 - label_smoothing) + 0.5 * label_smoothing)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits)
            mask = self.mask
            losses *= mask
            losses = tf.reduce_sum(losses, axis = 1)
            total_size = tf.reduce_sum(mask, axis = 1)
            losses = tf.divide(losses, total_size)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_sentence2(self, l2_lambda=0.001):
        # multi_class_labels: [batch_size, max_num_sequence]
        # logits: [batch_size, max_num_sequence]
        # losses: [batch_size, max_num_sequence]
        self.loss_compute()
        with tf.name_scope("loss_sentence"):
            losses = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.input_y1, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_word(self, l2_lambda=0.001):
        # logits:  [batch_size, sequence_length, document_length]
        # targets: [batch_size, sequence_length]
        # weights: [batch_size, sequence_length]
        # loss:     scalar
        with tf.name_scope("loss_word"):
            loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,targets=self.input_y2,weights=self.mask,average_across_timesteps=True,average_across_batch=True)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.decay_rate, staircase=True)
        self.learning_rate = learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_gradients)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op