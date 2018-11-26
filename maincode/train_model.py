# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from data_utils import *
from textsum_model import Neuralmodel
from gensim.models import KeyedVectors
from rouge import Rouge
import os
import math
import pickle
from tqdm import tqdm

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("tra_data_path","../src/neuralsum/dailymail/tra/","path of training data.")
tf.app.flags.DEFINE_string("tst_data_path","../src/neuralsum/dailymail/tst/","path of test data.")
tf.app.flags.DEFINE_string("val_data_path","../src/neuralsum/dailymail/val/","path of validation data.")
tf.app.flags.DEFINE_string("entity_path","../cache/entity_dict.pkl", "path of entity data.")
tf.app.flags.DEFINE_string("vocab_path","../cache/vocab","path of vocab frequency list")
tf.app.flags.DEFINE_integer("vocab_size",200000,"maximum vocab size.")

tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("decay_step", 3000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.1, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","../ckpt/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("embed_size", 150,"embedding size")
tf.app.flags.DEFINE_integer("input_y2_max_length", 40,"the max length of a sentence in abstracts")
tf.app.flags.DEFINE_integer("max_num_sequence", 70,"the max number of sequence in documents")
tf.app.flags.DEFINE_integer("sequence_length", 125,"the max length of a sentence in documents")
tf.app.flags.DEFINE_integer("hidden_size", 300,"the hidden size of the encoder and decoder")
tf.app.flags.DEFINE_integer("document_length", 1000,"the max vocabulary of documents")
tf.app.flags.DEFINE_integer("beam_width", 4,"the beam search max width")
tf.app.flags.DEFINE_integer("attention_size", 150,"the attention size of the decoder")
tf.app.flags.DEFINE_boolean("extract_sentence_flag", True,"using sentence extractor")
tf.app.flags.DEFINE_boolean("is_training", True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","../w2v/benchmark_sg1_e150_b.vector","word2vec's vocabulary and vectors")
filter_sizes = [1,2,3,4,5,6,7]
feature_map = [50,50,50,50,50,25,25]

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # instantiate model
        Model = Neuralmodel(FLAGS.extract_sentence_flag, FLAGS.vocab_size, FLAGS.batch_size, FLAGS.embed_size, FLAGS.learning_rate, FLAGS.decay_step, FLAGS.decay_rate, FLAGS.max_num_sequence, FLAGS.sequence_length,
                            filter_sizes, feature_map, FLAGS.hidden_size, FLAGS.document_length, FLAGS.beam_width, FLAGS.attention_size, FLAGS.input_y2_max_length)
        # initialize saver
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, FLAGS.vocab_path, FLAGS.entity_path, FLAGS.vocab_size, Model,FLAGS.word2vec_model_path)
        curr_epoch=sess.run(Model.epoch_step)

        batch_size=FLAGS.batch_size
        iteration=0
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, counter =  0.0, 0
            train_gen = Batch(FLAGS.tra_data_path,FLAGS.vocab_path,FLAGS.entity_path,FLAGS.batch_size,FLAGS)
            for batch in tqdm(train_gen):
                iteration=iteration+1
                if epoch==0 and counter==0:
                    print("train_batch", batch['article_words'])
                feed_dict={}
                feed_dict[Model.dropout_keep_prob] = 0.5
                feed_dict[Model.input_x] = batch['article_words']
                feed_dict[Model.input_y1] = batch['label_sentences']
                feed_dict[Model.input_y1_length] = batch['article_len']
                feed_dict[Model.is_training] = True
                curr_loss,lr,_,_=sess.run([Model.loss_val,Model.learning_rate,Model.train_op,Model.global_increment],feed_dict)
                loss,counter=loss+curr_loss,counter+1
                if counter %50==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))
                if iteration % 300 == 0: # eval every 3000 steps.
                    eval_loss, acc_score = do_eval(sess, Model)
                    print("Epoch %d Validation Loss:%.3f\t Acc:%.3f" % (epoch, eval_loss, acc_score))
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)
            print("going to increment epoch counter....")
            sess.run(Model.epoch_increment)
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                # TODO eval_loss,f1_score=do_eval(sess,Model,testX,testY,iteration,num_classes)
                # TODO print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f" % (epoch,eval_loss,f1_score))
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)
        # TODO test_loss,f1_score = do_eval(sess, Model, testX, testY, iteration, num_classes)
        # TODO print("Test Loss:%.3f" % (test_loss))

def do_eval(sess, Model):
    eval_loss, eval_counter, acc_score= 0.0, 0, 0.0
    batch_size = 20
    valid_gen = Batch(FLAGS.val_data_path,FLAGS.vocab_path,FLAGS.entity_path,batch_size,FLAGS)
    for batch in valid_gen:
        feed_dict={}
        feed_dict[Model.dropout_keep_prob] = 1.0
        feed_dict[Model.input_x] = batch['article_words']
        feed_dict[Model.input_y1] = batch['label_sentences']
        feed_dict[Model.input_y1_length] = batch['article_len']
        feed_dict[Model.is_training] = False
        curr_eval_loss,logits=sess.run([Model.loss_val,Model.logits],feed_dict)
        curr_acc_score = compute_rouge(logits, batch)
        acc_score += curr_acc_score
        eval_loss += curr_eval_loss
        eval_counter += 1

    return eval_loss/float(eval_counter), acc_score/float(eval_counter)

def compute_rouge(logits, batch):

    # from logits
    imp_pos = np.argsort(logits)
    abs_num = [ len(abst) for abst in batch['abstracts_len']]
    sen_pos = [ pos[:num] for pos, num in zip(imp_pos, abs_num) ]

    # from corpus
    rouge = Rouge()
    rou_pos = []
    for data_dict in batch['original']:
        article = data_dict['article']
        abstracts = data_dict['abstract']
        rou_pos_ = []
        for abstract in abstracts:
            mark = 0
            best = 0.0
            for pos, sentence in enumerate(article):
                scores = rouge.get_scores(hyps = abstract, refs = sentence)
                f1 = scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']
                if f1 > best: best = f1 ; mark = pos
            rou_pos_.append(mark)
        rou_pos.append(rou_pos_)

    # compute
    acc_list = []
    for sen, rou in zip(sen_pos, rou_pos):
        sen = set(sen)
        rou = set(rou)
        acc = 2.0 - len(sen|rou) / float(len(sen))
        acc_list.append(acc)
    acc_score = np.mean(acc_list)

    return acc_score

def assign_pretrained_word_embedding(sess,vocab_path,entity_path,vocab_size,Model,word2vec_model_path):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    vocab = Vocab(vocab_path, entity_path, vocab_size)
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size, dtype=np.float32)  # assign empty for first word:'PAD'
    for i in range(1, vocab_size):  # loop each word
        word = vocab.id2word(i)
        embedding = None
        try:
            embedding = word2vec_model[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(Model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)

    word_embedding_2dlist_ = [[]] * 2  # create an empty word_embedding list for GO END.
    word_embedding_2dlist_[0] = np.random.uniform(-bound, bound, FLAGS.hidden_size) # GO
    word_embedding_2dlist_[1] = np.random.uniform(-bound, bound, FLAGS.hidden_size) # END
    word_embedding_final_ = np.array(word_embedding_2dlist_)  # covert to 2d array.
    word_embedding_ = tf.constant(word_embedding_final_, dtype=tf.float32)  # convert to tensor
    t_assign_embedding_ = tf.assign(Model.Embedding_,word_embedding_)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding_)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

if __name__ == "__main__":
    tf.app.run()
