import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import sys

L0 = 1100.0
dt = 1e-3
nfeatures = int(sys.argv[1])


with tf.compat.v1.Session() as sess:	    
	with gfile.FastGFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.compat.v1.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.compat.v1.import_graph_def(graph_def)
		inputnodename  = graph_def.node[0].name
		outputnodename = graph_def.node[-1].name
		print(inputnodename)
		print(outputnodename)
		tensor_input = sess.graph.get_tensor_by_name('import/'+inputnodename+':0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/'+outputnodename+':0')	        	                    	   

df = pd.read_csv('../data/input_iso.csv')
input_n = df['n']

df_in = df.iloc[:, :nfeatures]
test_sample = df_in.to_numpy()

# calculate velocity
# test_sample[:,-1] = (test_sample[:,-1] - test_sample[:,-2])*(L0/dt)

with tf.compat.v1.Session() as sess:
  predictions = sess.run(tensor_output, {tensor_input:test_sample})	        
df['pb_prediction'] = predictions[:,0]
df.to_csv('../results/prediction.csv', index=False)

# Calculate the correlation coefficient
correlation_coefficient = input_n.corr(df['pb_prediction'])
print("Correlation Coefficient:", correlation_coefficient)
