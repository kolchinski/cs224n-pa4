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


class QASepSystem(qa_model.QASystem):

    def __init__(self, encoder, decoder, *args):
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = FLAGS.max_length

        #TODO: Define question max_length and ctx max_length


        # ==== set up placeholder tokens ========
        self.ques_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.ctx_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.max_length))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
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
