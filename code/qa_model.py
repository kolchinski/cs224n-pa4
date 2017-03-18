from __future__ import absolute_import, division, print_function

import random
import time
import logging
import os

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from util import Progbar
from evaluate import exact_match_score, new_f1_score

logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS


class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        embed_path = FLAGS.embed_path or os.path.join(
            "data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
        with open(embed_path, "rb") as f:
            self.pretrained_embeddings = np.load(f)['glove']

        self.boundary_token = np.random.randn(1, FLAGS.embedding_size)
        self.pretrained_embeddings = np.append(self.pretrained_embeddings, self.boundary_token, axis=0)
        self.boundary_token_index = len(self.pretrained_embeddings) - 1

        # We now need to set up the tensorflow emedding
        embed = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        embeddings = tf.nn.embedding_lookup(embed, self.input_placeholder)
        # embeddings = tf.reshape(extracted, (-1, self.max_length, FLAGS.embed_size))
        return embeddings

    def eval_sentence(self, preds_ind, gold_ind, sentence):
        pred_vecs = [s for s, p in zip(sentence, preds_ind) if p]
        gold_vecs = [s for s, g in zip(sentence, gold_ind) if g]

        pred_sent = ' '.join(self.vocab[i] for i in pred_vecs)
        gold_sent = ' '.join(self.vocab[i] for i in gold_vecs)

        f1 = new_f1_score(pred_sent, gold_sent)
        em = exact_match_score(pred_sent, gold_sent)
        return f1, em, pred_sent, gold_sent

    def train_on_batch(self, session, batch_data):
        """Perform one step of gradient descent on the provided batch of data.
        """

        inputs_batch, labels_batch, context_spans_batch = batch_data
        seq_lengths = [e + 1 for (s, e) in context_spans_batch]
        masks = [[0] * s + [1] * (e - s + 1) + [0] * (FLAGS.max_length - e - 1)
            for (s,e) in context_spans_batch]
        masks = np.array(masks)

        feed_dict = {self.input_placeholder: inputs_batch,
                     self.labels_placeholder: labels_batch,
                     self.seq_lengths_placeholder: seq_lengths,
                     self.mask_placeholder: masks,
                     self.dropout_placeholder: FLAGS.dropout}
        _, l = session.run([self.train_op, self.loss], feed_dict=feed_dict)

        return l

    def run_epoch(self, sess, train):
        batches = self.build_batches(self.train_qas)
        if not FLAGS.is_prod: batches = batches[:5]
        prog = Progbar(target=len(batches))
        losses = []
        for i, batch in enumerate(batches):
            loss = self.train_on_batch(sess, zip(*batch))
            losses.append(loss)
            prog.update(i + 1, [("train loss", loss)])
        f1, em = self.evaluate_answer(sess, log=True)

        return f1

    def fit(self, sess, saver, train, best_train_dir):
        f1s = []
        for epoch in range(FLAGS.epochs):
            self.epoch = epoch
            logging.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
            f1 = self.run_epoch(sess, train)
            saver.save(sess, FLAGS.output_path)
            f1_max = max(f1s) if len(f1s) > 0 else 0
            if f1 > f1_max:
                import shutil
                best_dir = best_train_dir + "/best"
                if os.path.exists(best_dir): shutil.rmtree(best_dir)
                shutil.copytree(FLAGS.output_path, best_dir)
                # also copy to the local best dir
                local_best_dir = FLAGS.output_path + "/best"
                if os.path.exists(local_best_dir): shutil.rmtree(local_best_dir)
                shutil.copytree(FLAGS.output_path, local_best_dir)
            f1s.append(f1)
        return f1s

    def process_dataset(self, dataset):
        all_contexts = dataset['contexts']
        all_questions = dataset['questions']
        all_spans = dataset['spans']
        self.vocab = dataset['vocab']
        self.vocab.append("<SEP>")
        assert(len(self.vocab) == len(self.pretrained_embeddings))

        self.train_contexts = all_contexts
        self.train_questions = all_questions
        self.train_spans = all_spans

        #print(train_spans[:2])

        all_seqs = map(lambda x,y: x + [self.boundary_token_index] + y +
                                     [0] * (FLAGS.max_length - len(x) - len(y) - 1),
                         all_questions, all_contexts)
        all_seqs = [x for x in all_seqs if len(x) <= FLAGS.max_length]

        def pad_spans(q, start, end):
            q_pad = len(q) + 1
            return self.selector_sequence(q_pad + start, q_pad + end, FLAGS.max_length)

        padded_spans = [pad_spans(q, start, end) for q, c, (start, end)
                        in zip(all_questions, all_contexts, all_spans)
                        if len(q) + len(c) + 1 <= FLAGS.max_length]

        seq_starts = [len(q) + 1 for (q,c) in zip(all_questions, all_contexts)
                      if len(q) + 1 + len(c) <= FLAGS.max_length]
        seq_ends = [len(q) + len(c)  for (q,c) in zip(all_questions, all_contexts)
                    if len(q) + 1 + len(c) <= FLAGS.max_length]
        seq_spans = zip(seq_starts, seq_ends)

        all_qs = list(zip(all_seqs, padded_spans, seq_spans))
        # random.shuffle(all_qs)  # change the train/validation set from run to run

        train_size = int(len(all_qs) * .8)
        self.train_qas = all_qs[:train_size]
        self.dev_qas = all_qs[train_size:]

    @staticmethod
    def selector_sequence(start, end, total_len):
        """
        :param start: The first element to make 1
        :param end: The last element to make 1 (can be the same as the start_1)
        :return:
        """
        if end < start:
            end = start
        center = end + 1 - start
        post = total_len - (end + 1)

        data = [0] * start + [1] * center + [0] * post
        return data

    def build_batches(self, qas_set, shuffle=True):
        """
        :param qas_set:  list of [question, seq]
        :return: batched lists of [question, seq]
        """
        # random.shuffle(qas_set)  # make different batches each time
        import math

        batch_size = FLAGS.batch_size
        num_batches = int(math.ceil(len(qas_set) / batch_size))
        print(num_batches, "batch_size: ", batch_size)

        batches = [qas_set[b_num * batch_size: (b_num + 1) * batch_size]
                   for b_num in range(num_batches)]

        if shuffle:
            random.shuffle(batches)

        return batches

    def train(self, session, best_t_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        initializer = tf.global_variables_initializer()
        session.run(initializer)
        print("Session initialized, starting training")

        print("Start train function")
        losses = self.fit(session, self.saver, self.train_qas, best_t_dir)

