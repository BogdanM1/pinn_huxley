import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import sys
import math

nfeatures = int(sys.argv[1])
''' fixed parameters ''' 
f1_0 = 43.3 
h = 15.6
g1 = 10.0
g2 = 208.0
fzah = 1.0
L0 = 1100.0
dt = 1e-3
grdstretch = [0.6, 0.8, 0.95, 1.0, 1.64, 5.0]
grdstress = [0.0, 0.782, 1.0, 1.0, 0.0, 0.0]
# sarcomere length
s=1.1


def NAnalytical(x, v):

  if(x>h): return 0

  V=2*v/s
  
  phi = (f1_0 + g1) * (h / s)
  term1 = f1_0 / (f1_0 + g1)
  term2 = 1 - math.exp(-phi / V)
  term3 = math.exp(2 * g2 * (x / (s * V)))  
  
  if(x<0):return (term1*term2*term3)
  
  term2 = (x*x)/(h*h)-1 
  term3 = phi/V
  
  return (term1*(1-math.exp(term2*term3)))

with tf.compat.v1.Session() as sess:	    
	with gfile.FastGFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.compat.v1.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.compat.v1.import_graph_def(graph_def)
		inputnodename  = graph_def.node[0].name
		outputnodename = graph_def.node[-1].name
		print("############ input and output node names:")
		print(inputnodename)
		print(outputnodename)
		print("#########################################")
		tensor_input = sess.graph.get_tensor_by_name('import/'+inputnodename+':0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/'+outputnodename+':0')	        	                    	   

df = pd.read_csv('../data/test.csv')
df_in = df.iloc[:, :nfeatures]
test_sample = df_in.to_numpy()
with tf.compat.v1.Session() as sess:
  predictions = sess.run(tensor_output, {tensor_input:test_sample})	
df['prediction'] = predictions[:,0]
df['nanalytical'] = df.apply(lambda row: NAnalytical(row['x'], row['v']), axis=1)
df.to_csv('../results/output.csv', index=False)
# Calculate the correlation coefficient
correlation_coefficient = df['nanalytical'].corr(df['prediction'])
print("Correlation Coefficient:", correlation_coefficient)
