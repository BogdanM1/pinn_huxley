import numpy as np
import random
import tensorflow as tf
import pandas as pd 
import sciann as sn
from sciann.utils.math import diff, sign
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_addons as tfa

''' fixed parameters ''' 
TOL = 1e-4
f1_0 = 43.3 
h = 15.6
g1 = 10.0
g2 = 208.0
fzah = 1.0
L0 = 1100.0
dt = 1e-3
grdstretch = [0.6, 0.8, 0.95, 1.0, 1.64, 5.0]
grdstress = [0.0, 0.782, 1.0, 1.0, 0.0, 0.0]
''' fixed parameters '''

def lininterp(x, x0, x1, y0, y1):
  return (y0 + (x-x0)*(y1 - y0)/(x1 - x0))

def gordon_correction(stretch, n):
    cor = ( (stretch >= grdstretch[0] and stretch < grdstretch[1])*lininterp(stretch, grdstretch[0],grdstretch[1], grdstress[0],grdstress[1]) 
        +   (stretch >= grdstretch[1] and stretch < grdstretch[2])*lininterp(stretch, grdstretch[1],grdstretch[2], grdstress[1],grdstress[2])
        +   (stretch >= grdstretch[2] and stretch < grdstretch[3])*lininterp(stretch, grdstretch[2],grdstretch[3], grdstress[2],grdstress[3])
        +   (stretch >= grdstretch[3] and stretch < grdstretch[4])*lininterp(stretch, grdstretch[3],grdstretch[4], grdstress[3],grdstress[4])
        +   (stretch >= grdstretch[4] and stretch < grdstretch[5])*lininterp(stretch, grdstretch[4],grdstretch[5], grdstress[4],grdstress[5]))
    return (cor > n)*(cor - n)

def f(x,a):
    return (x >= .0 and x <= h)*(f1_0*a*x/h)

def g(x):
    return ((x < .0)*g2 + 
            (x >= .0 and x <= h)*(g1*x/h) + 
            (x > h)*(fzah*g1*x/h))
      

x = sn.Variable('x')
t = sn.Variable('t')
a = sn.Variable('a')
stretch = sn.Variable('stretch')
stretch_prev = sn.Variable('stretch_prev')

n = sn.Functional('n', [x,t,a,stretch,stretch_prev], 8*[20], 'tanh')    
L1 = (diff(n, t) + (stretch-stretch_prev)*(L0/dt) * diff(n, x) - gordon_correction(stretch,n)*f(x,a) + n*g(x))*(1+sign(n)) * 0.5 
#L1 = (diff(n, t) + (stretch-stretch_prev)*(L0/dt) * diff(n, x) - (1-n)*f(x,a) + n*g(x))* (1+sign(n)) * 0.5 
I1 = (t < TOL )*n
I2 = (1-sign(n))*n
D1 = sn.Data(n)

model = sn.SciModel([x,t,a,stretch,stretch_prev], [L1*1e-3, I1, I2, D1])


df = pd.read_csv('../data/pinn_data_train.csv')
df = df.drop_duplicates()
x_train = np.array(df['x'])
t_train = np.array(df['t'])
a_train = np.array(df['activation'])
stretch_train = np.array(df['stretch'])
stretch_prev_train = np.array(df['stretch_prev'])
n_train = np.array(df['n'])  

nzeros = (int)(len(n_train)/10)
t_train = np.append(t_train, np.zeros(nzeros))
n_train = np.append(n_train, np.zeros(nzeros))

x_train = np.append(x_train, np.random.choice(x_train, size=nzeros, replace=False))
a_train = np.append(a_train, np.random.choice(a_train, size=nzeros, replace=False))
stretch_train = np.append(stretch_train, np.random.choice(stretch_train, size=nzeros, replace=False))
stretch_prev_train = np.append(stretch_prev_train, np.random.choice(stretch_prev_train, size=nzeros, replace=False))

'''
x_train = np.append(x_train, np.random.choice(x_train, size=nzeros, replace=False))
a_train = np.append(a_train, np.zeros(nzeros))
stretch_train = np.append(stretch_train, np.ones(nzeros))
stretch_prev_train = np.append(stretch_prev_train, np.ones(nzeros))

'''

#sample_weights =np.array([1.0 if (val>1e-10) else 100 for val in n_train])
#sample_weights *=np.array([1.0 if (val<1e-2 or val > h) else 100 for val in x_train])
h = model.train([x_train, t_train, a_train, stretch_train, stretch_prev_train], ['zeros', 'zeros', 'zeros', n_train], 
                 #weights=sample_weights, 
                 learning_rate=1e-5, batch_size=16384, epochs=9000, verbose=2,
                 save_weights = {'path':'../models/best_model', 'best':True,'freq':2},
                 adaptive_weights={'method':'NTK', 'freq':400})

