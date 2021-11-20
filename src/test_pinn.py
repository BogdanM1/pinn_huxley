import numpy as np 
import sys
from keras import backend as K
from keras.models import  Sequential
from keras.layers import Dense
    
model_path    = '../models/model.hdf5'
model = Sequential()
model.add(Dense(20, input_dim = 2, activation='tanh'))
for i in range(7):
  model.add(Dense(20, activation='tanh'))
model.add(Dense(1))
model.load_weights(model_path)
model.compile(loss='mse', optimizer='adam')
    

x_test = np.arange(-20.8,63,5.2)
t_test = np.array([0,0.001,0.002, 0.4])

test_sample = []
for t_val in t_test:
  for x_val in x_test: 
    test_sample += [[x_val,t_val]]
    
prediction = model.predict(np.array(test_sample))
original_stdout = sys.stdout 
with open('../results/prediction.csv', 'w') as f:
  sys.stdout = f 
  print('t,x,n')
  index = 0
  for t_val in t_test:
    for x_val in x_test: 
      print(t_val,',',x_val,',',prediction[index][0]) 
      index += 1
  sys.stdout = original_stdout