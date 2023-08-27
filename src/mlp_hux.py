from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import  Sequential
from numpy.random import seed
import tensorflow as tf
import pandas as pd
import numpy as np 

_seed = 137
seed(_seed)
tf.random.set_seed(_seed)

model_path    = '../models/model-mlp.h5'


model = Sequential()
model.add(Dense(200, input_dim = 5, activation='tanh'))
model.add(Dropout(0.1))
for i in range(7):
  model.add(Dense(200, activation='tanh'))
  model.add(Dropout(0.1))
model.add(Dense(2))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))


df = pd.read_csv('../data/pinn_data_train.csv')
x_train = np.array(df['x'])
t_train = np.array(df['t'])
a_train = np.array(df['activation'])
stretch_train = np.array(df['stretch'])
stretch_prev_train = np.array(df['stretch_prev'])
n_train = np.array(df['n'])  

Input = np.column_stack((x_train, t_train, a_train, stretch_train, stretch_prev_train))

df = pd.read_csv('../data/pinn_data_val.csv')
x_val = np.array(df['x'])
t_val = np.array(df['t'])
a_val = np.array(df['activation'])
stretch_val = np.array(df['stretch'])
stretch_prev_val = np.array(df['stretch_prev'])
n_val = np.array(df['n']) 
Input_val = np.column_stack((x_val, t_val, a_val, stretch_val, stretch_prev_val)) 

history = model.fit(Input, n_train, batch_size=8192, epochs=10000, verbose=2, validation_data=(Input_val,n_val),
                    callbacks=[ModelCheckpoint(model_path, save_best_only=True)])