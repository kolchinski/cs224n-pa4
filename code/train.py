# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

# from qa_model import Encoder, QASystem, Decoder
from qa_sep_model import QASepSystem
from os.path import join as pjoin

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)


tf.app.flags.DEFINE_float("recall_multiplier", 100, "Proportionality multiplier for false negative penalty")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 100, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("output_path", "results/{:%Y%m%d_%H%M%S}/".format(datetime.now()), "output locations")

tf.app.flags.DEFINE_string("max_length", 250, "Length of longest context we'll use")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    try:
        os.symlink(os.path.abspath(train_dir), global_train_dir)
    except Exception:
        pass
    return global_train_dir


def load_data_file(data_dir, name):
    with open(os.path.join(data_dir, name)) as f:
        return [[int(w) for w in l.strip().split()] for l in f.readlines()]


def main(_):
    #embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    os.chdir('..')
    # we never use this
    # print("Initializing vocab")
    # vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    # vocab, rev_vocab = initialize_vocab(vocab_path)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    print("Building QASystem")
    qa = QASepSystem(FLAGS.embedding_size, FLAGS.state_size, FLAGS.output_size)

    dataset = load_dataset(FLAGS.data_dir)
    qa.process_dataset(dataset, max_c_length=300, max_q_length=30)
    qa.build_pipeline()

    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print("Flags: ", vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        print("Initializing model")
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess)

        print("Evaluating answer")
        qa.evaluate_answer(sess, dataset['vocab'], FLAGS.evaluate, log=True)


def load_dataset(data_dir):
    print("Loading training data")
    with open(os.path.join(data_dir, 'vocab.dat'), "rb") as f:
        vocab = [str(l).strip() for l in f.readlines()]

    dataset = {}
    dataset['contexts'] = load_data_file(data_dir, 'train.ids.context')
    dataset['questions'] = load_data_file(data_dir, 'train.ids.question')
    dataset['spans'] = load_data_file(data_dir, 'train.span')
    dataset['vocab'] = vocab
    return dataset


if __name__ == "__main__":
    tf.app.run()
