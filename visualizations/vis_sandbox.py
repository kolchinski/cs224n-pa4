import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from scipy.stats import gaussian_kde
import seaborn as sns

import pandas as pd


loaded = np.load('best_epoch.npz') 
s = loaded['start_probs']
e = loaded['end_probs']
g = loaded['gold_spans']

n = len(g)
l = s.shape[1]


starts = np.argmax(s, axis=1)
ends = np.argmax(e, axis=1)
pred_lengths = ends - starts + 1
x = pred_lengths = np.clip(pred_lengths, 0, None)
y = true_lengths = g[:,1] - g[:,0] + 1
sns.jointplot(x,y, kind='scatter', xlim=(0,40), ylim=(0,30))
xy = np.vstack([x,y])
z = cmap(gaussian_kde(xy)(xy) * 2000)
fig, ax = plt.subplots()
tmp = ax.scatter(x, y, c=z, s=100, edgecolor='')
#sns.heatmap(x,y)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Answer span lengths")
plt.colorbar(tmp)
plt.show()

'''
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy) * 2000
fig, ax = plt.subplots()
tmp = ax.scatter(x, y, c=z, s=100, edgecolor='')
plt.xlim(-1,30)
plt.ylim(-1,30)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Answer span lengths")
plt.colorbar(tmp)
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'r-')
plt.show()
'''

lengths_corr = pd.DataFrame()





bins = np.linspace(0, 1, 20)
topStartProbs = np.max(s, axis=1)
topEndProbs = np.max(e, axis=1)

#plt.hist(topStartProbs, bins, alpha=0.5, label='Start index', color='crimson')
#plt.hist(topEndProbs, bins, alpha=0.5, label='End index', color='blue')
#plt.title("Probability of chosen start/end index")
#plt.legend(loc='upper left')

#startProbDiffs = []
#endProbDiffs = []
#for i in range(len(s)):
#  tmp_s = np.sort(s[i])
#  startProbDiffs.append(tmp_s[-1] - tmp_s[-2])
#  tmp_e = np.sort(e[i])
#  endProbDiffs.append(tmp_e[-1] - tmp_e[-2])
#
#
##plt.hist(startProbDiffs, bins, alpha=0.5, label='Start index', color='crimson')
##plt.hist(endProbDiffs, bins, alpha=0.5, label='End index', color='blue')
#sns.distplot(startProbDiffs, bins, label='Start index', kde=False, color='crimson')
#sns.distplot(endProbDiffs, bins, label='End index', kde=False, color='blue')
#plt.title("Probability margin of chosen start/end index over next likeliest")
#plt.legend(loc='upper right')
#plt.show()



