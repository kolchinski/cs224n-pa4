from __future__ import absolute_import, division, print_function

import logging
import random
import os

import tensorflow as tf
import numpy as np

import multiprocessing

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
        with tf.variable_scope("encoder") as scope:
            q_outputs, q_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell, cell_bw=cell,
                sequence_length=q_lens, inputs=qs, dtype=tf.float32,
                swap_memory=True)

            q_states_fw, q_states_bw = q_states

            #Keep the same parameters for encoding questions and contexts
            scope.reuse_variables()

            #Run the BiLSTM on the contexts, starting with the hidden states from the questions
            c_outputs, c_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell, cell_bw=cell,
                sequence_length=c_lens, inputs=cs, dtype=tf.float32,
                initial_state_bw=q_states_bw, initial_state_fw=q_states_fw,
                swap_memory=True)

            c_outputs_fw, c_outputs_bw = c_outputs
            c_states_fw, c_states_bw = c_states

        return q_outputs, q_states, c_outputs, c_states


# Take in the output of the BiEncoder and turn it into a vector of probabilities
class AttentionBiDecoder(object):
    def __init__(self, output_size, hidden_size):
        self.output_size = output_size
        self.hidden_size = hidden_size

    def decode(self, init_state, q_embeds, c_embeds, input_lens, masks, dropout):
        init_state_fw, init_state_bw = init_state

        # TODO: Fix this, should be doing something better than just adding the fwd and backwd encodings
        # q_embeds = q_embeds[0] + q_embeds[1]
        # c_embeds = c_embeds[0] + c_embeds[1]
        q_embeds_fw, q_embeds_bw = q_embeds
        q_embeds = tf.concat(2, [q_embeds_fw, q_embeds_bw])

        c_embeds_fw, c_embeds_bw = c_embeds
        c_embeds = tf.concat(2, [c_embeds_fw, c_embeds_bw])

        # inputs = c_embeds[0] + c_embeds[1]

        with vs.variable_scope("decoder") as scope:
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, use_peepholes=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - dropout)
            #Run the decoder LSTM on the outputs of the forward encoder
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell, cell_bw=cell,
                sequence_length=input_lens, dtype=tf.float32, inputs=c_embeds,
                initial_state_bw=init_state_bw, initial_state_fw=init_state_fw,
                swap_memory=True
                )
            outputs_fw, outputs_bw = outputs
            states_fw, states_bw = states


        # Attention mechanism code follows (implemented as post-decoder global attention initially)
        # For reference, see paper at http://nlp.stanford.edu/pubs/emnlp15_attn.pdf
        # and post at https://piazza.com/class/iw9g8b9yxp46s8?cid=2106

        #H states from encoder LSTMs (concatenated),
        # dimensions (batch size) x (question length + context length) x (embedding size)
        qc_embed = tf.concat(1, [q_embeds, c_embeds])

        #H-states from decoder LSTM, dimensions (batch size) x (context length) x (2*embedding size)
        decodings = tf.concat(2, [outputs_fw, outputs_bw])

        xav_init = tf.contrib.layers.xavier_initializer()
        #Compute the bilinear product of encoder and decoder outputs to generate the attention weights vector
        # Final dimensions: (batch size) x (context length) x (question length + context length)
        w_a = tf.get_variable("W_a", (2*self.hidden_size, 2*self.hidden_size), tf.float32, xav_init)
        #This would be the right way to do it but this use of einsum is not supported in tf yet :(
        #pairwise_scores = tf.einsum('bnd,dd,bmd->bmn', qc_embed, w_a, decodings)
        #instead, proceed stepwise
        #step one produces a (batch size) x (hidden size) x (context length + question length)) matrix
        m1 = tf.matmul(tf.reshape(qc_embed,[-1, 2*self.hidden_size]), w_a)
        m1 = tf.reshape(m1, [-1, 2*self.hidden_size, tf.shape(qc_embed)[1]])

        #Now, multiply the (b x d x n) matrix by a (b x m x d) matrix to get a (b x m x n) matrix
        pairwise_scores = tf.matmul(decodings, m1)

        #Now, apply softmax to get the actual attention weights - softmax automatically normalizes over the last dim
        #Shape same as pairwise_scores
        attn_weights = tf.nn.softmax(pairwise_scores)

        #Now, use the above-calculated weights to calculate a weighted average of encoder outputs for each
        #position of the decoder output. Output shape: (batch size) x (context length) x (embedding size)
        weighted_embeddings = tf.einsum('bmn,bnd->bmd', attn_weights, qc_embed)

        #Append the attention-ified encoder embedding with our decoder embedding
        #Output shape: (batch_size) x (context length) x (2*embedding size)
        extended_states = tf.concat(2, [decodings, weighted_embeddings])

        #Multiply the new state through by a matrix to convert it to a new "attentionified state"
        #Output shape: (batch size) x (context length) x (embedding size)
        w_c = tf.get_variable("W_c", (4*self.hidden_size, self.hidden_size), tf.float32, xav_init)
        #h_tilde = tf.einsum('bme,ed->bmd', extended_states, w_c)
        m1 = tf.reshape(extended_states, [-1, 4*self.hidden_size])
        #h_tilde = tf.reshape(tf.matmul(m1, w_c), [-1, tf.shape(c_embeds)[0], self.hidden_size])
        #Don't reshape to lead into the next op more easily
        h_tilde = tf.matmul(m1, w_c)

        #Finally, multiply the newly transformed states each by a final transformation vector to get the
        #predicted probability that the word at a given position of a given batch is part of the answer
        #Final output shape: (batch size) x (context length)
        w_s = tf.get_variable("W_s", (self.hidden_size, 1), tf.float32, xav_init)
        word_res = tf.reshape(tf.matmul(h_tilde,w_s), [tf.shape(c_embeds)[0], -1])
        #word_res = tf.nn.sigmoid(word_res)

        #zero out irrelevant positions (before and after context) of predictions
        word_res = word_res * masks
        return word_res

#Take in the output of the BiEncoder and turn it into a vector of probabilities
class NaiveBiDecoder(object):
    def __init__(self, output_size, hidden_size):
        self.output_size = output_size
        self.hidden_size = hidden_size



    def decode(self, init_state, inputs, input_lens, masks, dropout):
        init_state_fw, init_state_bw = init_state
        inputs_fw, inputs_bw = inputs
        #Stack forward and backward input vectors along embedding dimension
        inputs = tf.concat(2, [inputs_fw, inputs_bw])


        with vs.variable_scope("decoder"):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, use_peepholes=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - dropout)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell, cell_bw=cell,
                sequence_length=input_lens, dtype=tf.float32, inputs=inputs,
                initial_state_bw=init_state_bw, initial_state_fw=init_state_fw,
                swap_memory=True )
            outputs_fw, outputs_bw = outputs


        xav_init = tf.contrib.layers.xavier_initializer()

        #output two values instead of 1? for positive and negative class
        #then run through softmax
        w = tf.get_variable("W_final", (2*self.hidden_size, 1), tf.float32, xav_init)
        b = tf.get_variable("b_final", (1,), tf.float32, tf.constant_initializer(0.0))

        word_res = tf.concat(2, [outputs_fw, outputs_bw])

        word_res = tf.reshape(word_res, [-1, 2*self.hidden_size])
        inner = tf.matmul(word_res, w) + b
        #use relu and softmax here instead?
        #inner = tf.nn.sigmoid(inner)
        word_res = tf.reshape(inner, [-1, self.output_size])

        #zero out irrelevant positions (before and after context) of predictions
        word_res = word_res * masks
        return word_res

    def coattention(self, q_embeddings, c_embeddings):
        q_embed_transpose = tf.transpose(q_embedings, perm=[0, 2, 1])
        # computing unnormalized affinity matrix
        unnorm_affinity = tf.batch_matmul(c_embeddings, q_embed_transpose) #batch_size, context, question
        # shape = (batch, question, context)
        unnorm_affinity_t = tf.transpose(unnorm_affinity, perm=[0,2,1])
        # compute row-wise normalized wrt question (dimensions: context, question)
        norm_wrt_q = tf.map_fn(lambda x: tf.nn.softmax(x),unnorm_affinity_t,dtype=tf.float32)
        # compute col-wise wrt context (dimensions: question, context)
        norm_wrt_c = tf.map_fn(lambda x: tf.nn.softmax(x),unnorm_affinity,dtype=tf.float32)
        # summaries wrt question
        summ_wrt_q = tf.batch_matmul(norm_wrt_q, c_embeddings)
        summ_q_embed = tf.concat(1, [q_embed_transpose, tf.transpose(summ_wrt_q, perm=[0, 2 ,1])])
        # summaries of previous attention with respect to context
        summ_wrt_c = tf.batch_matmul(summ_q_embed, norm_wrt_c, adj_y=True)
        # final coattention context, (batch_size, context+1, 3*hidden_size)
        co_att = tf.concat(2, [c_embeddings, tf.transpose(summ_wrt_c, perm=[0, 2, 1])])

        #LSTM for encoding
        cell_forward = tf.nn.rnn_cell.LSTMCell(hidden_size)
        cell_backward = tf.nn.rnn_cell.LSTMCell(hidden_size)
        # compute coattention encoding
        
        # u, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, co_att, sequence_length=tf.to_int64([max_timesteps]*batch_size),dtype=tf.float32) 
        # self._u = tf.concat(2, u)


class QASepSystem(qa_model.QASystem):
    def __init__(self, input_size, hidden_size, output_size, *args):
        self.in_size = input_size
        self.hidden_size = hidden_size
        # self.out_size = output_size
        self.eval_res_file = open(FLAGS.log_dir + "/eval_res.txt")

    def build_pipeline(self):
        # ==== set up placeholder tokens ========
        # Question and context sequences
        self.encoder = BiEncoder(self.hidden_size, self.in_size)
        #self.decoder = NaiveBiDecoder(self.max_c_len, self.hidden_size)
        self.decoder = AttentionBiDecoder(self.max_c_len, self.hidden_size)

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
        # def encode(self, qs, q_lens, cs, c_lens, dropout):
        hidden_rep = self.encoder.encode(embeds["q"], self.q_len_pholder, embeds["ctx"],
                                         self.c_len_pholder, self.dropout_placeholder)
        q_out, q_state, c_out, c_states = hidden_rep
        # res = self.decoder.decode(q_state, c_out, self.c_len_pholder, self.mask_placeholder,
        #                          self.dropout_placeholder)
        res = self.decoder.decode(q_state, q_out, c_out, self.c_len_pholder, self.mask_placeholder,
                                  self.dropout_placeholder)
        return res

    def decode_arbitration_layer(self, word_res, masks):
        # If we are doing masking, we should also mask before this.
        # that way the nn gets an accurate assessment of the actual probs
        masked_wr = word_res * masks
        res1_inner = self.simple_arb_layer(masked_wr, "arb_layer_1")
        res1 = tf.nn.relu(res1_inner)
        res2_inner = self.simple_arb_layer(res1, "arb_layer_2")
        res2 = tf.nn.relu(res2_inner)  # we might not want this relu layer
        masked_res = res2 * masks
        return masked_res

    def simple_arb_layer(self, inputs, layer_name):
        with vs.variable_scope(layer_name):
            xav_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable("W_arb", [self.max_c_len, self.max_c_len], tf.float32, xav_init)
            b = tf.get_variable("B_arb", [self.max_c_len], tf.float32, tf.constant_initializer(0.0))
            inner = tf.matmul(inputs, w) + b
        return inner

    def process_dataset(self, dataset, max_q_length=None, max_c_length=None):
        self.train_contexts = all_cs = dataset['contexts']
        self.train_questions = all_qs = dataset['questions']
        self.train_spans = all_spans = dataset['spans']
        self.vocab = dataset['vocab']

        self.max_q_len = max_q_length or max(all_qs, key=len)
        self.max_c_len = max_c_length or max(all_cs, key=len)

        # build the padded questions, contexts, spans, lengths
        pad_qs, pad_cs, pad_spans, seq_lens = (list() for i in range(4))

        for q, c, span in zip(all_qs, all_cs, all_spans):
            if len(q) > max_q_length:
                continue
            if len(c) > max_c_length:
                continue
            pad_qs.append(self.pad_ele(q, self.max_q_len))
            pad_cs.append(self.pad_ele(c, self.max_c_len))
            start, end = span
            pad_spans.append(self.selector_sequence(start, end, self.max_c_len))
            seq_lens.append((len(q), len(c)))

        # now we sort the whole thing
        all_qs = list(zip(pad_qs, pad_cs, pad_spans, seq_lens))
        train_size = int(len(all_qs) * .8)
        self.train_qas = all_qs[:train_size]
        self.dev_qas = all_qs[train_size:]

        sort_alg = lambda x: x[3][1] + x[3][0] / 1000  # small bias for quesiton length
        self.train_qas.sort(key=sort_alg)
        self.dev_qas.sort(key=sort_alg)


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
        return [QASepSystem.pad_ele(s, max_len) for s in seqs], max_len

    @staticmethod
    def pad_ele(seq, max_len):
        return seq + (max_len - len(seq)) * [0]

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
        eval_set = list(random.sample(self.dev_qas, sample))
        q_vec, ctx_vec, gold_spans, masks = zip(*eval_set)

        pred_probs = []
        for batch in self.build_batches(eval_set, shuffle=False):
            feed_dict = self.prepare_data(zip(*batch))
            pred_probs.extend(session.run([self.results], feed_dict=feed_dict))

        pred_spans = [[int(m > 0.5) for m in n] for n in pred_probs]

        f1s, ems, pred_s, gold_s = zip(*(self.eval_sentence(p, g, s)
                         for p, g, s in zip(pred_spans, gold_spans, ctx_vec)))

        f1 = np.mean(f1s)
        em = np.mean(ems)

        if log:
            logging.info("\nF1: {}, EM: {}, for {} samples".format(f1, em, sample))
            logging.info("{} mean prob; {} total words predicted".format(
                np.mean(pred_probs), np.sum(pred_spans)))

            # all the evaluate info
            text = lambda vecs: ' '.join(self.vocab[i] for i in vecs)

            # sorting into buckets
            em_sents, partial_matches, no_match, empty = [[] for i in range(4)]
            for ques, gold, our, is_em, sample_f1 in zip(q_vec, gold_s, pred_s, ems, f1s):
                if len(our) == 0:
                    empty.append((ques, gold))
                elif is_em:
                    em_sents.append((ques, gold))
                elif sample_f1 > 0:
                    partial_matches.append((ques, gold, our))
                else:
                    no_match.append((ques, gold, our))

            self.eval_res_file.write("\n\nEpoch {}:".format(self.epoch))

            # Yes, my fellow CS107 TAs will hate this....
            if len(em_sents):
                self.eval_res_file.write("Sample Exact Matches")
                for ques, gold in em_sents[:5]:
                    self.eval_res_file.write("Ques: " + text(ques))
                    self.eval_res_file.write("Answ: " + gold)

            if len(empty):
                self.eval_res_file.write("Sents where we didn't predict anything")
                for ques, gold in empty[:5]:
                    self.eval_res_file.write("Ques: " + text(ques))
                    self.eval_res_file.write("Answ: " + gold)

            if len(partial_matches):
                self.eval_res_file.write("Partial matches")
                for ques, gold, our in partial_matches[:5]:
                    self.eval_res_file.write("Ques: " + text(ques))
                    self.eval_res_file.write("Answ: " + gold)
                    self.eval_res_file.write("OurA: " + our)

            if len(no_match):
                self.eval_res_file.write("Partial matches")
                for ques, gold, our in no_match[:5]:
                    self.eval_res_file.write("Ques: " + text(ques))
                    self.eval_res_file.write("Answ: " + gold)
                    self.eval_res_file.write("OurA: " + our)

        return f1, em

