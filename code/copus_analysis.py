from __future__ import print_function
import IPython

import train
import numpy as np
import pprint

dataset = train.load_dataset('data/squad')


all_cs = dataset['contexts']
all_qs = dataset['questions']
all_spans = dataset['spans']
vocab = dataset['vocab']

q_len = [len(q) for q in all_qs]
c_len = [len(c) for c in all_cs]
span_len = [e - s + 1 for s, e in all_spans]

def hist(data, bins):
    np_hist = np.histogram(data, bins)
    pprint.pprint(list(zip(*np_hist)))


print("ques hist: ")
hist(q_len, 20)

print("ctx hist: ",)
hist(c_len, 20)

print("span_end hist: ", )
hist(list(zip(*all_spans))[1], 20)


print("avg span_len: ", np.mean(span_len))
print("span_len hist: ",)
hist(span_len, 20)


# start interactive mode
IPython.embed()


