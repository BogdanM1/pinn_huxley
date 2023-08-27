import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

with tf.compat.v1.Session() as sess:	    
	with gfile.FastGFile('../../models/model-isom.pb', 'rb') as f:	        
		graph_def = tf.compat.v1.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.compat.v1.import_graph_def(graph_def)	        
		tensor_input = sess.graph.get_tensor_by_name('import/dense_input:0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/dense_8/BiasAdd:0')	        	                    	   

model_path    = '../../models/isom-best_model-best.hdf5'
model = Sequential()
model.add(Dense(20, input_dim = 2, activation='tanh'))
for i in range(7):
  model.add(Dense(20, activation='tanh'))
model.add(Dense(1))
model.load_weights(model_path)
model.compile(loss='mse', optimizer='adam')


df = pd.read_csv('../../data/original-isom.csv')
df = df.iloc[:, :-1]
df = df.iloc[:, 0:2]
test_sample = df.to_numpy()
with tf.compat.v1.Session() as sess:
  predictions = sess.run(tensor_output, {tensor_input:test_sample})	        
df['pb_prediction'] = predictions[:,0]
output = model.predict(test_sample)
df['hdf5_prediction'] = output[:,0]
df.to_csv('../../results/prediction-isom.csv', index=False)
