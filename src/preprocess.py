import pandas as pd
import numpy as np
import itertools

col_names = ['x','t','activation','stretch','stretch_prev','n']
ntrains = 1
ntraine = 11

for i in range(ntrains,ntraine,1):
	df = pd.read_csv('../data/experiments/experiment'+str(i)+'.csv', names=col_names)
	df = df.drop_duplicates()
	if(i==1):
		df.to_csv('../data/pinn_data_train.csv', index=False)
	else:
		df.to_csv('../data/pinn_data_train.csv', mode='a', header=False, index=False)		

for i in itertools.chain(range(ntrains+3,ntraine,4)):
	df = pd.read_csv('../data/experiments/experiment'+str(i)+'.csv', names=col_names)
	df = df.drop_duplicates()
	if(i==4):
		df.to_csv('../data/pinn_data_val.csv', index=False)
	else:
		df.to_csv('../data/pinn_data_val.csv', mode='a', header=False, index=False)
