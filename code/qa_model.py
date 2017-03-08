from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import logging

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

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

    def encode(self, inputs, encoder_state_input):
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
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        print(cell.state_size)
        #print(encoder_state_input.get_shape())
        word_res, f_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        #word_res, f_state = tf.nn.dynamic_rnn(cell, inputs,
        #                    initial_state = encoder_state_input)
        return f_state, word_res


class Decoder(object):
    def __init__(self, output_size, hidden_size):
        self.output_size = output_size
        self.hidden_size = hidden_size

    def decode(self, knowledge_rep):
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
            word_res, _ = tf.nn.dynamic_rnn(cell, encode_out, initial_state=encode_fstate)

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
        return  word_res



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
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, ())


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            embeds = self.setup_embeddings()
            self.results = self.setup_system(embeds)
            loss = self.setup_loss(self.results)

        # ==== set up training/updating procedure ====
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    def setup_system(self, embeds):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        initial_hidden_c = tf.placeholder(tf.float32, (None, FLAGS.state_size))
        initial_hidden_h = tf.placeholder(tf.float32, (None, FLAGS.state_size))
        initial_hidden = tf.nn.rnn_cell.LSTMStateTuple(initial_hidden_c, initial_hidden_h)
        hidden_rep = self.encoder.encode(embeds, initial_hidden)
        res = self.decoder.decode(hidden_rep)
        return res

    def setup_loss(self, final_res):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            y = self.labels_placeholder
            losses = tf.nn.l2_loss((final_res - y))
            loss = tf.reduce_mean(losses)
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

        self.boundary_token = np.random.randn(FLAGS.embedding_size)
        np.append(self.pretrained_embeddings, self.boundary_token)

        self.boundary_token_index = len(self.boundary_token) - 1

        # We now need to set up the tensorflow emedding

        embed = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        embeddings = tf.nn.embedding_lookup(embed, self.input_placeholder)
        # embeddings = tf.reshape(extracted, (-1, self.max_length, FLAGS.embed_size))
        ### END YOUR CODE
        return embeddings


    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed['train_x'] = train_x
        input_feed['train_y'] = train_y
        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.results]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance

        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
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

        print("Start train function")

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        #TODO: Batch up data, run loop over batches
        train_contexts = dataset['train_contexts']
        train_questions = dataset['train_questions']
        train_spans = dataset['train_spans']
        print(train_spans[:2])

        train_seqs = map(lambda x,y: x +
                        [self.boundary_token] + y, train_contexts, train_questions)

        padded_spans = [[0] * (len(q) + 1 + start)+ [1] *(end + 1 - start) + [0] * (len(c) - end - 1)
                        for q, c, (start, end)
                        in zip(train_questions, train_contexts, train_spans)]

        padded_spans_2 = []
        n = len(train_contexts)
        assert (n == len(train_questions))
        for i in range(n):
            s, e = train_spans[i]
            curSeq = [0] * len(train_questions[i])  # question padding
            curSeq += [0]  # for the separator token
            curSeq += [0] * s + [1] * (e - s + 1)  # first 0s, 1s for span
            curSeq += [0] * (len(train_contexts[i]) - e - 1)  # 0s after span
            padded_spans_2.append(curSeq)

        assert (padded_spans == padded_spans_2)

        #  batch the data
        batch_size = FLAGS.batch_size
        print("with length", len(train_contexts), "type", type(len(train_contexts)))
        num_batches = len(train_contexts)//batch_size
	print(num_batches, "batch_size: ", batch_size)

        all_qs = list(zip(train_seqs,padded_spans))
	random.shuffle(all_qs)
	print(type(all_qs))
        batches = [all_qs[b_num * batch_size: (b_num + 1)*batch_size]
                   for b_num in range(num_batches)]

        initializer = tf.global_variables_initializer()
        session.run(initializer)
        print("Session initialized, starting training")

        for b_num, b in enumerate(batches):
            if b_num % 100 == 0:
                print("Training on batch #{}".format(b))

            ques_con_seq, labels = zip(*b)
            feed_dict = {self.input_placeholder: ques_con_seq,
                         self.labels_placeholder: labels}
            session.run(self.train_op, feed_dict=feed_dict)



