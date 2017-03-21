import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm


def plotSpanDensity(s, e, g=[47,52]):
  assert(s.shape == e.shape)
  l = len(s)
  #s = np.sum(starts, axis = 1)
  x = np.empty([10,l]) 
  x[:,:] = s 
  plt.contourf(x, l, cmap=mpl.cm.jet) 
  plt.show()



loaded = np.load('best_epoch.npz') 
s = loaded['start_probs']
e = loaded['end_probs']
g = loaded['gold_spans']

n = len(g)
l = s.shape[1]


fake_start = 45
fake_end = 55
fake_middle = 50.0
fake_length = (fake_end - fake_start)/2.0
s_mass = np.zeros(100)
e_mass = np.zeros(100)

for i in range(n):
  true_start, true_end = g[i]
  true_length = (true_end - true_start) / 2.
  true_middle = (true_start + true_end) / 2.

  for j in range(l):
    lengths_from_mean = (j - true_middle) / (true_length + 1)
    fake_index = int(fake_middle + lengths_from_mean*fake_length)
    if fake_index >=0 and fake_index <100:
      s_mass[fake_index] += s[i,j]
      e_mass[fake_index] += e[i,j]

  


