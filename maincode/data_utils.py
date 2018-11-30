# -*- coding:utf-8 -*-
import numpy as np
import pickle
import random
import os
from tflearn.data_utils import pad_sequences

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def Batch(data_path, vocab_path, entity_path, size, hps):

    res = {}
    filenames = os.listdir(data_path)
    random.shuffle(filenames)
    label_sentences, article_value, article_words, article_len, abstracts_targets, abstracts_inputs, abstracts_len, results = [], [], [], [], [], [], [], []
    vocab = Vocab(vocab_path, entity_path, hps.vocab_size)
    for cnt, filename in enumerate(filenames):
        pickle_path = os.path.join(data_path, filename)
        res = load(pickle_path)
        label, value, words, len_a, targets, inputs, lens = Example(res['article'],res['abstract'],res['label'],res['entity'], vocab, hps) # TODO
        label_sentences.append(label)
        article_value.append(value)
        article_words.append(words)
        article_len.append(len_a)
        abstracts_targets.append(targets)
        abstracts_inputs.append(inputs)
        abstracts_len.append(lens)
        results.append(res)

        if (cnt+1) % size == 0 and cnt != 0:
            data_dict ={}
            data_dict['label_sentences'] = label_sentences
            data_dict['article_value'] = article_value
            data_dict['article_words'] = article_words
            data_dict['article_len'] = article_len
            data_dict['abstracts_targets'] = abstracts_targets
            data_dict['abstracts_inputs'] = abstracts_inputs
            data_dict['abstracts_len'] = abstracts_len
            data_dict['original'] = results
            label_sentences, article_value, article_words, article_len, abstracts_targets, abstracts_inputs, abstracts_len, results = [], [], [], [], [], [], [], []
            yield data_dict

def Example(article, abstracts, label, entity, vocab, hps):

    # get ids of special tokens
    start_decoding = vocab.word2id(START_DECODING)
    stop_decoding = vocab.word2id(STOP_DECODING)
    pad_id = vocab.word2id(PAD_TOKEN)

    """process the label"""
    # pos 2 multi one-hot
    label_sentences = label2ids(label, hps.max_num_sequence)

    """process the article"""
    # create vocab and word 2 id
    article_value = value2ids(article, vocab, hps.document_length)
    # word 2 id
    article_words = article2ids(article, vocab)
    # num sentence
    article_len = len(article)
    # word level padding
    article_words = pad_sequences(article_words, maxlen=hps.sequence_length, value=pad_id)
    # sentence level padding
    pad_article = np.expand_dims(np.zeros(hps.sequence_length, dtype=np.int32), axis = 0)
    if article_words.shape[0] > hps.max_num_sequence:
        article_words = article_words[:hps.max_num_sequence]
    while article_words.shape[0] < hps.max_num_sequence:
        article_words = np.concatenate((article_words, pad_article))

    """process the abstract"""
    # word 2 id
    abstracts_words = abstract2ids(abstracts, vocab)
    # add tokens
    abstracts_inputs, abstracts_targets = token2add(abstracts_words, hps.input_y2_max_length, start_decoding, stop_decoding)
    # search id in value position
    abstract_targets = value2pos(abstracts_targets, article_words)
    # padding
    abstracts_inputs = pad_sequences(abstracts_inputs, maxlen=hps.input_y2_max_length, value=pad_id)
    abstracts_targets = pad_sequences(abstracts_targets, maxlen=hps.input_y2_max_length, value=pad_id)
    # sentence level padding
    pad_abstracts = np.expand_dims(np.zeros(hps.input_y2_max_length, dtype=np.int32), axis = 0)
    if abstracts_inputs.shape[0] > hps.max_num_abstract:
        abstracts_inputs = abstracts_inputs[:hps.max_num_abstract]
    while abstracts_inputs.shape[0] < hps.max_num_abstract:
        abstracts_inputs = np.concatenate((abstracts_inputs, pad_abstracts))
    if abstracts_targets.shape[0] > hps.max_num_abstract:
        abstracts_targets = abstracts_targets[:hps.max_num_abstract]
    while abstracts_inputs.shape[0] < hps.max_num_abstract:
        abstracts_targets = np.concatenate((abstracts_targets, pad_abstracts))
    # mask
    abstracts_len = abstract2len(abstracts)

    return label_sentences, article_value, article_words, article_len, abstracts_targets, abstracts_inputs, abstracts_len

class Vocab(object):
    def __init__(self, vocab_file, entity_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    continue
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN,START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is'% w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    break

        with open(entity_file, 'rb') as output:
            data = pickle.load(output)
            for key,value in data.items():
                self._word_to_id[key] = self._count
                self._id_to_word[self._count] = key
                self._count += 1

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        #if word_id not in self._id_to_word:
        #    raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

def label2ids(labels, label_size):
    res = np.zeros(label_size, dtype=np.int32)
    label_list = [ pos for pos in labels if pos < label_size]
    res[label_list] = 1
    return res

def value2ids(article, vocab, document_length):
    value = []
    pad_id = vocab.word2id(PAD_TOKEN)
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    stop_id = vocab.word2id(STOP_DECODING)
    value.append(unk_id)
    value.append(stop_id)
    for sent in article:
        article_words = sent.split()
        for w in article_words:
            i = vocab.word2id(w)
            if i == unk_id:
                pass
            if i not in value:
                value.append(i)
    cnt = 4
    while len(value) < document_length:
        if cnt not in value:
            value.append(cnt)
        cnt += 1
    return value

def value2pos(abstract, value):
    poss = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for sent in abstract:
        pos=[]
        for i in sent:
            if i in value:
                pos.append(value.index(i))
            else:
                pos.append(value.index(unk_id))
        poss.append(pos)
    return poss

def article2ids(article, vocab):
    idss = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for sent in article:
        ids = []
        article_words = sent.split()
        for w in article_words:
            i = vocab.word2id(w)
            if i == unk_id:
                if w not in oovs:
                    oovs.append(w)
                ids.append(i)
            else:
                ids.append(i)
        idss.append(ids)
    return idss

def abstract2ids(abstracts, vocab):
    idss= []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for sent in abstracts:
        ids = []
        abstract_words = sent.split()
        for w in abstract_words:
            i = vocab.word2id(w)
            if i == unk_id:
                ids.append(i)
            else:
                ids.append(i)
        idss.append(ids)
    return idss

def token2add(abstracts, max_len, start_id, stop_id):
    inps = []
    targets = []
    for sequence in abstracts:
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:
            inp = inp[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)
        assert len(inp) == len(target)
        inps.append(inp)
        targets.append(target)
    return inps, targets

def abstract2len(abstracts):
    length = []
    for sent in abstracts:
        abstract_words = sent.split()
        length.append(len(abstract_words)+1)
    return length

def outputids2words(id_list, vocab):
    words = []
    for i in id_list:
        w = vocab.id2word(i)
        words.append(w)
    return words

