from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
from os.path import join as pjoin
import six

from tqdm import tqdm
import tensorflow as tf

from qa_sep_model import QASepSystem
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging
logging.basicConfig(level=logging.INFO)

import socket
is_azure = (socket.gethostname() == "cs224n-gpu")


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.6, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_string("is_prod", is_azure, "Adjust batch size and num of epochs for non prod for debugging")
# tf.app.flags.DEFINE_string("output_path", "results/{:%Y%m%d_%H%M%S}/".format(datetime.now()), "output locations")

def initialize_model(session, model, train_dir, is_eval=False):
    ckpt = tf.train.get_checkpoint_state(train_dir + "/best/")
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #logging.info("Current dir {}".format(os.listdir('./tmp/')))
        #model.saver.restore(session, ckpt.model_checkpoint_path)
        model.saver.restore(session, '../tmp/cs224n-squad-train/best/')
    else:
        if is_eval:
            raise Exception("Couldn't find model parameters for eval. ckpt: {}")
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
        #raise ValueError("Vocabulary file %s not found.", vocab_path)
        raise ValueError("Vocabulary file {} not found. Current dir {}".format(vocab_path, os.listdir('.')))


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []
    all_context_tokens = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = list(tokenize(context))

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [vocab.get(str(w), qa_data.UNK_ID) for w in context_tokens]
                qustion_ids = [vocab.get(str(w), qa_data.UNK_ID) for w in question_tokens]

                # context_data.append(' '.join(context_ids))
                # query_data.append(' '.join(qustion_ids))
                context_data.append(context_ids)
                query_data.append(qustion_ids)
                question_uuid_data.append(question_uuid)
                all_context_tokens.append(context_tokens)

    return context_data, all_context_tokens, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, context_tokens, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, context_tokens, question_uuid_data


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
    try:  # to make it work on windows
        os.symlink(os.path.abspath(train_dir), global_train_dir)
    except Exception:
        return train_dir
    return global_train_dir


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, context_tokens, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)

    dataset = {"contexts": context_data, "questions": question_data,
               "q_uuids":question_uuid_data, "vocab": vocab}

    # dataset = load_dataset(FLAGS.data_dir)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    truncated = True

    qa = QASepSystem(FLAGS.embedding_size, FLAGS.state_size)

    if truncated:
        eval_ds = qa.process_eval_dataset(dataset, max_c_length=300, max_q_length=30)
    else:
        eval_ds = qa.process_eval_dataset(dataset)
    qa.build_pipeline()

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        #if not is_azure: os.chdir("..")
        initialize_model(sess, qa, train_dir, True)
        answers = qa.gen_test_answers(sess, eval_ds, rev_vocab, context_tokens)

        print(os.listdir('.'))
        print(os.getcwd())
        print(os.path.abspath('.'))

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(six.text_type(json.dumps(answers, ensure_ascii=False, indent=2)))


if __name__ == "__main__":
  tf.app.run()
