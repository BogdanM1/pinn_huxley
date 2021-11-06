import numpy as np 
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.models import  Sequential
from keras.layers import Dense
from keras import backend as K

t_test = np.array([0,0.001,0.002])
x_test = np.arange(-20.8,62.4,5.2)
test_sample = []
for t_val in t_test:
  for x_val in x_test: 
    test_sample += [[t_val,x_val]]
    
test_sample = np.array(test_sample) 


def print_predictions(predictions):
    print('t,x,n')
    index = 0
    for t_val in t_test:
      for x_val in x_test: 
        print(t_val,',',x_val,',',predictions[index][0]) 
        index += 1   

with tf.compat.v1.Session() as sess:	    
	with gfile.FastGFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.compat.v1.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.compat.v1.import_graph_def(graph_def)	        
		tensor_input = sess.graph.get_tensor_by_name('import/dense_input:0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/dense_8/BiasAdd:0')	        
		predictions = sess.run(tensor_output, {tensor_input:test_sample})	        
		print_predictions(predictions)    


model_path    = '../models/model.hdf5'
K.set_learning_phase(0)	

model = Sequential()
model.add(Dense(100, input_dim = 2, activation='tanh'))
for i in range(7):
  model.add(Dense(100, activation='tanh'))
model.add(Dense(1))
model.load_weights(model_path)
model.compile(loss='mse', optimizer='adam')
output = model.predict(test_sample)
print_predictions(output)

