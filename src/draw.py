import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv('../data/pinn_data_train.csv')

plt.figure(figsize=(5, 4), dpi=300)
plt.hist([x for x in data['n']], bins=100, histtype='stepfilled', alpha=0.3, ec='k', color='rebeccapurple')
plt.savefig('hist_full.png')
plt.close() 