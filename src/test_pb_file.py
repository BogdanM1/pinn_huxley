import numpy as np 
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.models import  Sequential
from keras.layers import Dense
from keras import backend as K

sample = np.array( [
                    [0, 1, 0, 0],
                    [ 0, 1, 0, 0],
                    [0, 1, 0, 0]
                    ] )

with tf.compat.v1.Session() as sess:	    
	with gfile.FastGFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.compat.v1.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.compat.v1.import_graph_def(graph_def)	        
		tensor_input = sess.graph.get_tensor_by_name('import/dense_input:0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/dense_8/BiasAdd:0')	        
		predictions = sess.run(tensor_output, {tensor_input:sample})	        
		print(predictions)	        


model_path    = '../models/model.hdf5'
K.set_learning_phase(0)	

model = Sequential()
model.add(Dense(20, input_dim = 4, activation='tanh'))
for i in range(7):
  model.add(Dense(20, activation='tanh'))
model.add(Dense(1))
model.load_weights(model_path)
model.compile(loss='mse', optimizer='adam')
output = model.predict(sample)
print(output)

