from __future__ import absolute_import, division, print_function

import logging
import random
import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS

import qa_model

"""
This file is for the version of the QA_Model with separate
Questions and answer input placeholders.
"""


#Take in questions and contexts batch-wise
#Feed questions through one LSTM, then contexts through another
class BiEncoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, qs, q_lens, cs, c_lens, dropout):
        cell = tf.nn.rnn_cell.LSTMCell(self.size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - dropout)

        #Run the first BiLSTM on the questions
        q_outputs, q_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            dtype=tf.float32,
            sequence_length=q_lens,
            inputs=qs)

        q_states_fw, q_states_bw = q_states


        #Run the second BiLSTM on the contexts, starting with the hidden states from the question BiLSTM
        c_outputs, c_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            dtype=tf.float32,
            sequence_length=c_lens,
            inputs=cs,
            initial_state_bw=q_states_bw,
            initial_state_fw=q_states_fw)

        c_outputs_fw, c_outputs_bw = c_outputs
        c_states_fw, c_states_bw = c_states

        return q_outputs, c_outputs





class QASepSystem(qa_model.QASystem):

    def __init__(self, encoder, decoder, *args):
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = FLAGS.max_length

        #TODO: Define question max_length and ctx max_length


        # ==== set up placeholder tokens ========
        #Question and context sequences
        self.q_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.ctx_placeholder = tf.placeholder(tf.int32, (None, self.max_length))

        #Lengths of those sequences
        self.q_lengths_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.c_lengths_placeholder = tf.placeholder(tf.int32, (None, self.max_length))

        #True 1/0 labelings of words in the context
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.max_length))

        #1/0 mask to ignore padding in context for loss purposes
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))

        #Proportion of connections to drop
        self.dropout_placeholder = tf.placeholder(tf.float32, ())


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            embeds = self.setup_embeddings()
            self.results = self.setup_system(embeds)
            self.loss = self.setup_loss(self.results)

        # ==== set up training/updating procedure ====
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def process_dataset(self, dataset, max_q_length=None, max_c_length=None):
        self.train_contexts = all_cs = dataset['contexts']
        self.train_questions = all_qs = dataset['questions']
        self.train_spans = all_spans = dataset['spans']
        self.vocab = dataset['vocab']
        self.vocab.append("<SEP>")
        assert(len(self.vocab) == len(self.pretrained_embeddings))

        pad_qs = self.pad_vocab_ids(all_qs, max_q_length)
        pad_ctxs = self.pad_vocab_ids(all_cs, max_c_length)

        if max_c_length:
            all_spans = (s for s in all_spans if len(s) <= max_c_length)
        else:
            max_c_length = len(pad_ctxs[0])  # because the are already all max len

        padded_spans = [self.selector_sequence(start, end, max_c_length)
                        for start, end in all_spans]

        seq_ends = [len(c) for c in zip(all_cs)]

        all_qs = list(zip(pad_qs, pad_ctxs, padded_spans, seq_ends))
        # random.shuffle(all_qs)  # change the train/validation set from run to run

        train_size = int(len(all_qs) * .8)
        self.train_qas = all_qs[:train_size]
        self.dev_qas = all_qs[train_size:]

    @staticmethod
    def pad_vocab_ids(seqs, max_len=None):
        if max_len is None:
            max_len = max((len(s) for s in seqs))
        else:
            seqs = (s for s in seqs if len(s) <= max_len)
        return [s + (max_len - len(s)) * [0] for s in seqs]

    def evaluate_answer(self, session, sample=500, log=True):
        eval_set = random.sample(self.dev_qas, sample)
        q_vec, ctx_vec, gold_spans, masks = zip(*eval_set)

        feed_dict = {self.ques_placeholder: q_vec, self.ctx_placeholder: ctx_vec}
        pred_probs, = session.run([self.results], feed_dict=feed_dict)
        pred_spans = [[int(round(m)) for m in n] for n in pred_probs]

        f1s, ems = zip(*(self.eval_sentence(p, g, s)
                         for p, g, s in zip(ctx_vec, gold_spans, pred_spans)))

        f1 = np.mean(f1s)
        em = np.mean(ems)

        if log:
            logging.info("\nF1: {}, EM: {}, for {} samples".format(f1, em, sample))
            logging.info("{} mean prob; {} total words predicted".format(
                np.mean(pred_probs), np.sum(pred_spans)))

        return f1, em
