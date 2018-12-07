# -*- coding: utf-8 -*-
import re
import os
import math
import pickle
import codecs
import json
import tensorflow as tf
import numpy as np
from data_utils import *
from textsum_model import Neuralmodel
from gensim.models import KeyedVectors
from rouge import Rouge

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("result_path","../src/neuralsum/dailymail/tst/","path to store the predicted results.")
tf.app.flags.DEFINE_string("tst_data_path","../src/neuralsum/dailymail/tst/","path of test data.")
tf.app.flags.DEFINE_string("tst_file_path","../src/neuralsum/dailymail/tst/","file of test data.")
tf.app.flags.DEFINE_string("entity_path","../cache/entity_dict.pkl", "path of entity data.")
tf.app.flags.DEFINE_string("vocab_path","../cache/vocab","path of vocab frequency list")
tf.app.flags.DEFINE_integer("vocab_size",200000,"maximum vocab size.")

tf.app.flags.DEFINE_float("learning_rate",0.0001,"learning rate")

tf.app.flags.DEFINE_integer("is_frozen_step", 0, "how many steps before fine-tuning the embedding.")
tf.app.flags.DEFINE_integer("cur_learning_step", 10000, "how many steps before using the predicted labels instead of true labels.")
tf.app.flags.DEFINE_integer("decay_step", 5000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.1, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","../ckpt/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("embed_size", 150,"embedding size")
tf.app.flags.DEFINE_integer("input_y2_max_length", 40,"the max length of a sentence in abstracts")
tf.app.flags.DEFINE_integer("max_num_sequence", 30,"the max number of sequence in documents")
tf.app.flags.DEFINE_integer("max_num_abstract", 4,"the max number of abstract in documents")
tf.app.flags.DEFINE_integer("sequence_length", 100,"the max length of a sentence in documents")
tf.app.flags.DEFINE_integer("hidden_size", 300,"the hidden size of the encoder and decoder")
tf.app.flags.DEFINE_boolean("use_highway_flag", True,"using highway network or not.")
tf.app.flags.DEFINE_integer("highway_layers", 1,"How many layers in highway network.")
tf.app.flags.DEFINE_integer("document_length", 1000,"the max vocabulary of documents")
tf.app.flags.DEFINE_integer("beam_width", 4,"the beam search max width")
tf.app.flags.DEFINE_integer("attention_size", 150,"the attention size of the decoder")
tf.app.flags.DEFINE_boolean("extract_sentence_flag", True,"using sentence extractor")
tf.app.flags.DEFINE_boolean("is_training", False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","../w2v/benchmark_sg1_e150_b.vector","word2vec's vocabulary and vectors")
filter_sizes = [1,2,3,4,5,6,7]
feature_map = [20,20,30,40,50,70,70]

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def dump(filename, data):
    with open(filename, 'w') as output:
        json.dump(data, ouptut)

def main(_):
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        Model=Neuralmodel(FLAGS.extract_sentence_flag, FLAGS.is_training, FLAGS.vocab_size, FLAGS.batch_size, FLAGS.embed_size, FLAGS.learning_rate, cur_learning_steps, FLAGS.decay_step, FLAGS.decay_rate, FLAGS.max_num_sequence, FLAGS.sequence_length,
                            filter_sizes, feature_map, FLAGS.use_highway_flag, FLAGS.highway_layers, FLAGS.hidden_size, FLAGS.document_length, FLAGS.max_num_abstract, FLAGS.beam_width, FLAGS.attention_size, FLAGS.input_y2_max_length)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return

        predict_gen = Batch_F(process_file(FLAGS.tst_file_path, FLAGS.entity_path), FLAGS.vocab_path, FLAGS.entity_path, FLAGS)
        predict_gen = Batch_P(FLAGS.tst_data_path, FLAGS.vocab_path, FLAGS.entity_path, FLAGS)
        for batch in predicted_gen:
            feed_dict={}
            feed_dict[Model.dropout_keep_prob] = 1.0
            feed_dict[Model.input_x] = batch['article_words']
            #feed_dict[Model.input_y1] = batch['label_sentences']
            #feed_dict[Model.input_y1_length] = batch['article_len']
            feed_dict[Model.tst] = False
            feed_dict[Model.cur_learning] = False
            logits = sess.run(Model.logits, feed_dict=feed_dict)
            data = compute_score(logits, batch)

def process_file(data_path, entity_path):
    data = []
    entitys = load(entity_path)
    with codecs.open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            example = {}
            entity_dict = {}
            for idx, name in entitys.items():
                if re.search(name, line):
                    article = line.replace(name, idx)
                    entity_dict[idx] = name
            example['article'] = article.splits('.')
            example['entity'] = entity_dict
            data.append(example)
    return data

def compute_list(logits, batch):
    data = batch['original']
    score_list = []
    for sent, score in zip(data['article'], logits[0][:len(data['article'])]):
        score_list.append((sent, score))
    data['score'] = sorted(score_list, key=lambda x:x[1], reverse=True)
    return data

if __name__ == '__main__':
    tf.app.run()
