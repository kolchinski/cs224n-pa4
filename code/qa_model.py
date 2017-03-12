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
from evaluate import exact_match_score, f1_score


logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, seq_lengths, dropout, encoder_state_input = None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.


        Current implementation in progress: Feed in question, then token, then context

        """

        # input_p = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.embedding_size))
        cell = tf.nn.rnn_cell.LSTMCell(self.size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - dropout)
        #print(encoder_state_input.get_shape())
        word_res, f_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_lengths, dtype=tf.float32)
        #word_res, f_state = tf.nn.dynamic_rnn(cell, inputs,
        #                    initial_state = encoder_state_input)
        return f_state, word_res


class Decoder(object):
    def __init__(self, output_size, hidden_size):
        self.output_size = output_size
        self.hidden_size = hidden_size

    def decode(self, knowledge_rep, seq_lengths, masks, dropout):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        We are just predicting: what is part of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        encode_fstate, encode_out = knowledge_rep

        with vs.variable_scope("decoder"):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, use_peepholes=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - dropout)
            word_res, _ = tf.nn.dynamic_rnn(cell, encode_out, sequence_length=seq_lengths, initial_state=encode_fstate)

        # now I need a final classification layer
        # result is a vector that represents all outputs

        xav_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable("W_final", (self.hidden_size, 1), tf.float32, xav_init)
        b = tf.get_variable("b_final", (1,), tf.float32, tf.constant_initializer(0.0))

        word_res = tf.reshape(word_res, [-1, FLAGS.state_size])
        inner = tf.matmul(word_res, w) + b
        #inner = tf.einsum()
        word_res = tf.nn.sigmoid(inner)
        word_res = tf.reshape(word_res, [-1, FLAGS.max_length])

        #zero out irrelevant positions (before and after context) of predictions
        word_res = self.decode_arbitration_layer(word_res, masks)
        word_res = word_res * masks
        return word_res

    def decode_accum_layer(self, node_out):
        """
        :param node_out: Should be a 3d tensor [batch, num_nodes, word_rep]
        :return: prediction tensor [batch, num_nodes
        Consider a window based model.
        Other stuff: note that we already do a shit ton of convolution and pooling,
        another layer wouldn't be that useful.
        """

    def decode_arbitration_layer(self, word_res, masks):
        # If we are doing masking, we should also mask before this.
        # that way the nn gets an accurate assessment of the actual probs
        masked_wr = word_res * masks
        xav_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable("W_arb", [FLAGS.max_length, FLAGS.max_length], tf.float32, xav_init)
        b = tf.get_variable("B_arb", [FLAGS.max_length], tf.float32, tf.constant_initializer(0.0))
        inner = tf.matmul(masked_wr, w) + b
        return tf.nn.sigmoid(inner)



class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = FLAGS.max_length


        # ==== set up placeholder tokens ========
        self.input_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.max_length))
        self.seq_lengths_placeholder = tf.placeholder(tf.int32, (None))
        self.mask_placeholder = tf.placeholder(tf.float32, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, ())


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            embeds = self.setup_embeddings()
            self.results = self.setup_system(embeds)
            self.loss = self.setup_loss(self.results)

        # ==== set up training/updating procedure ====
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def setup_system(self, embeds):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        hidden_rep = self.encoder.encode(embeds, self.seq_lengths_placeholder, self.dropout_placeholder)
        res = self.decoder.decode(hidden_rep, self.seq_lengths_placeholder, self.mask_placeholder,
                                  self.dropout_placeholder)
        return res

    def setup_loss(self, final_res):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            y = self.labels_placeholder
            orig_loss = tf.nn.l2_loss((final_res - y))
            #previous_loss_metric = tf.reduce_mean(orig_loss)
            # now we need to weight the losses for missing the 1
            weighted_loss = (FLAGS.recall_multiplier * y + 1) * orig_loss
            loss = tf.reduce_mean(weighted_loss)
            #loss = tf.Print(loss, [loss / previous_loss_metric])

        return loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        #with vs.variable_scope("embeddings"):
        #with open(os.path.join(FLAGS.data_dir, 'glove.trimmed.100.npz')) as f:
        #self.pretrained_embeddings = np.load(f)

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
        ### END YOUR CODE
        return embeddings


    def evaluate_answer(self, session, sample=500, log=True):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        eval_set = random.sample(self.dev_qas, sample)
        sent_vec, gold_spans, context_spans = zip(*eval_set)

        seq_lengths = [e + 1 for (s,e) in context_spans]
        masks = [ [0] * s + [1] * (e - s + 1) + [0] * (FLAGS.max_length - e - 1)
            for (s,e) in context_spans]
        masks = np.array(masks)

        feed_dict = {self.input_placeholder: sent_vec,
                     self.seq_lengths_placeholder: seq_lengths,
                     self.mask_placeholder: masks,
                     self.dropout_placeholder: 0}

        pred_probs, = session.run([self.results], feed_dict=feed_dict)
        pred_spans = [[int(round(m)) for m in n] for n in pred_probs]
        # don't need to remask this.

        f1s, ems = zip(*(self.eval_sentence(p, g, s)
                         for p, g, s in zip(pred_spans, gold_spans, sent_vec)))

        f1 = np.mean(f1s)
        em = np.mean(ems)

        if log:
            logging.info("\nF1: {}, EM: {}, for {} samples".format(f1, em, sample))
            logging.info("{} mean prob; {} total words predicted".format(
                np.mean(pred_probs), np.sum(pred_spans)))

        return f1, em


    def eval_sentence(self, preds_ind, gold_ind, sentence):
        pred_vecs = [s for s, p in zip(sentence, preds_ind) if p]
        gold_vecs = [s for s, g in zip(sentence, gold_ind) if g]

        pred_sent = ' '.join(self.vocab[i] for i in pred_vecs)
        gold_sent = ' '.join(self.vocab[i] for i in gold_vecs)

        f1 = f1_score(pred_sent, gold_sent)
        em = exact_match_score(pred_sent, gold_sent)
        return f1, em

    def train_on_batch(self, session, batch_data):
        """Perform one step of gradient descent on the provided batch of data.
        """

        inputs_batch, labels_batch, context_spans_batch = batch_data
        seq_lengths = [e + 1 for (s, e) in context_spans_batch]
        masks = [ [0] * s + [1] * (e - s + 1) + [0] * (FLAGS.max_length - e - 1)
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
        prog = Progbar(target=1 + int(len(train) / FLAGS.batch_size))
        losses = []
        for i, batch in enumerate(self.build_batches(self.train_qas)):
            #ques_con_seq, labels = zip(*b)
            loss = self.train_on_batch(sess, zip(*batch))
            losses.append(loss)
            prog.update(i + 1, [("train loss", loss)])
        f1, em = self.evaluate_answer(sess, log=True)

        return losses

    def fit(self, sess, saver, train):
        losses = []
        for epoch in range(FLAGS.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
            loss = self.run_epoch(sess, train)
            saver.save(sess, FLAGS.output_path)
            losses.append(loss)
        return losses

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

    def build_batches(self, qas_set):
        """
        :param qas_set:  list of [question, seq]
        :return: batched lists of [question, seq]
        """
        random.shuffle(qas_set)  # make different batches each time

        batch_size = FLAGS.batch_size
        num_batches = len(qas_set)//batch_size
        print(num_batches, "batch_size: ", batch_size)

        batches = [qas_set[b_num * batch_size: (b_num + 1) * batch_size]
                   for b_num in range(num_batches)]

        return batches

    def train(self, session):
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
        saver = tf.train.Saver()
        session.run(initializer)
        print("Session initialized, starting training")

        print("Start train function")
        losses = self.fit(session, saver, self.train_qas)

