import pandas as pd
col_names = ['x','t','activation','stretch','stretch_prev','n']

for i in range(1,166,2):
	df = pd.read_csv('../data/experiments/experiment'+str(i)+'.csv', names=col_names)
	df = df.drop_duplicates()
	if(i==1):
		df.to_csv('../data/pinn_data1.csv', index=False)
	else:
		df.to_csv('../data/pinn_data1.csv', mode='a', header=False, index=False)		
