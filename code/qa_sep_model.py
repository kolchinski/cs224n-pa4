from __future__ import absolute_import, division, print_function

import logging
import random
import os

import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS

import qa_model
from tensorflow.python.ops import variable_scope as vs

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
        with tf.variable_scope("ques"):
            q_outputs, q_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell, cell_bw=cell,
                sequence_length=q_lens, inputs=qs, dtype=tf.float32,
                swap_memory=True)

        q_states_fw, q_states_bw = q_states

        #Run the second BiLSTM on the contexts, starting with the hidden states from the question BiLSTM
        with tf.variable_scope("c_en"):
            c_outputs, c_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell, cell_bw=cell,
                sequence_length=c_lens, inputs=cs, dtype=tf.float32,
                initial_state_bw=q_states_bw, initial_state_fw=q_states_fw,
                swap_memory=True)

            c_outputs_fw, c_outputs_bw = c_outputs
            c_states_fw, c_states_bw = c_states

        return q_outputs, q_states, c_outputs, c_states


#Take in the output of the BiEncoder and turn it into a vector of probabilities
class NaiveBiDecoder(object):
    def __init__(self, output_size, hidden_size):
        self.output_size = output_size
        self.hidden_size = hidden_size

    def decode(self, init_state, inputs, input_lens, masks, dropout):
        init_state_fw, init_state_bw = init_state
        max_len = tf.shape(inputs)[1]

        with vs.variable_scope("decoder"):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, use_peepholes=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - dropout)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell, cell_bw=cell,
                sequence_length=input_lens, dtype=tf.float32, inputs=inputs,
                initial_state_bw=init_state_bw, initial_state_fw=init_state_fw,
                swap_memory=True
                )
            outputs_fw, outputs_bw = outputs


        xav_init = tf.contrib.layers.xavier_initializer()

        w = tf.get_variable("W_final", (2*self.hidden_size, 1), tf.float32, xav_init)
        b = tf.get_variable("b_final", (1,), tf.float32, tf.constant_initializer(0.0))

        #Fix this - need to stack forward and backward outputs
        word_res_fw = tf.reshape(outputs_fw, [-1, self.hidden_size])
        word_res_bw = tf.reshape(outputs_bw, [-1, self.hidden_size])
        word_res = tf.concat(1, [word_res_fw, word_res_bw])

        inner = tf.matmul(word_res, w) + b
        word_res = tf.nn.sigmoid(inner)
        word_res = tf.reshape(word_res, [-1, self.output_size])

        #zero out irrelevant positions (before and after context) of predictions
        word_res = word_res * masks
        return word_res


class QASepSystem(qa_model.QASystem):
    def __init__(self, input_size, hidden_size, output_size, *args):
        self.in_size = input_size
        self.hidden_size = hidden_size
        # self.out_size = output_size

    def build_pipeline(self):
        # ==== set up placeholder tokens ========
        # Question and context sequences
        self.encoder = BiEncoder(self.hidden_size, self.in_size)
        self.decoder = NaiveBiDecoder(self.max_c_len, self.hidden_size)

        self.q_placeholder = tf.placeholder(tf.int32, (None, self.max_q_len))
        self.ctx_placeholder = tf.placeholder(tf.int32, (None, self.max_c_len))

        self.q_len_pholder = tf.placeholder(tf.int32, (None,))
        self.c_len_pholder = tf.placeholder(tf.int32, (None,))

        # True 1/0 labelings of words in the context
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.max_c_len))

        # 1/0 mask to ignore padding in context for loss purposes
        self.mask_placeholder = tf.placeholder(tf.float32, (None, self.max_c_len))

        # Proportion of connections to drop
        self.dropout_placeholder = tf.placeholder(tf.float32, ())

        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            embeds = self.setup_embeddings()
            self.results = self.setup_system(embeds)
            self.loss = self.setup_loss(self.results)

        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)


    def setup_embeddings(self):
        embed_path = FLAGS.embed_path or os.path.join(
            "data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
        with open(embed_path, "rb") as f:
            self.pretrained_embeddings = np.load(f)['glove']

        # We now need to set up the tensorflow emedding
        embed = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        q_embed = tf.nn.embedding_lookup(embed, self.q_placeholder)
        ctx_embed = tf.nn.embedding_lookup(embed, self.ctx_placeholder)

        return {"q": q_embed, "ctx": ctx_embed}

    def setup_system(self, embeds):
        #def encode(self, qs, q_lens, cs, c_lens, dropout):
        hidden_rep = self.encoder.encode(embeds["q"], self.q_len_pholder, embeds["ctx"],
                                         self.c_len_pholder, self.dropout_placeholder)
        q_out, q_state, c_out, c_states = hidden_rep
        res = self.decoder.decode(q_state, c_out, self.c_len_pholder, self.mask_placeholder,
                                  self.dropout_placeholder)
        return res

    def process_dataset(self, dataset, max_q_length=None, max_c_length=None):
        self.train_contexts = all_cs = dataset['contexts']
        self.train_questions = all_qs = dataset['questions']
        self.train_spans = all_spans = dataset['spans']
        self.vocab = dataset['vocab']

        pad_qs, self.max_q_len = self.pad_and_max_len(all_qs)
        pad_ctxs, self.max_c_len = self.pad_and_max_len(all_cs)

        if max_c_length:
            all_spans = (s for s in all_spans if len(s) <= max_c_length)
        else:
            max_c_length = len(pad_ctxs[0])  # because the are already all max len

        padded_spans = [self.selector_sequence(start, end, max_c_length)
                        for start, end in all_spans]

        seq_ends = [(len(q), len(c)) for q, c in zip(all_qs, all_cs)]
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

    @staticmethod
    def pad_and_max_len(seqs):
        max_len = max((len(s) for s in seqs))
        return [s + (max_len - len(s)) * [0] for s in seqs], max_len

    def train_on_batch(self, session, batch_data):
        """Perform one step of gradient descent on the provided batch of data.
        """

        feed_dict = self.prepare_data(batch_data, dropout=FLAGS.dropout)
        _, l = session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l

    def prepare_data(self, data, dropout=0):
        q_batch, ctx_batch, labels_batch, context_spans_batch = data
        q_lens, c_lens = zip(*context_spans_batch)
        masks = [self.selector_sequence(0, c - 1, self.max_c_len)  for c in c_lens]

        feed_dict = {self.q_placeholder: q_batch,
                     self.ctx_placeholder: ctx_batch,
                     self.labels_placeholder: labels_batch,
                     self.q_len_pholder: q_lens,
                     self.c_len_pholder: c_lens,
                     self.dropout_placeholder: dropout,
                     self.mask_placeholder: masks}
        return feed_dict

    def evaluate_answer(self, session, sample=500, log=True):
        eval_set = random.sample(self.dev_qas, sample)
        q_vec, ctx_vec, gold_spans, masks = zip(*eval_set)

        feed_dict = self.prepare_data(zip(*eval_set))
        pred_probs, = session.run([self.results], feed_dict=feed_dict)
        pred_spans = [[int(m > 0.5) for m in n] for n in pred_probs]

        f1s, ems = zip(*(self.eval_sentence(p, g, s)
                         for p, g, s in zip(pred_spans, gold_spans, ctx_vec)))

        f1 = np.mean(f1s)
        em = np.mean(ems)

        if log:
            logging.info("\nF1: {}, EM: {}, for {} samples".format(f1, em, sample))
            logging.info("{} mean prob; {} total words predicted".format(
                np.mean(pred_probs), np.sum(pred_spans)))

        return f1, em
