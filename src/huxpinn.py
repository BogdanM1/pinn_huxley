import numpy as np
import random
import tensorflow as tf
import pandas as pd 
import sciann as sn
from sciann.utils.math import diff, sign

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
LFACTOR = 0.1
''' fixed parameters '''

def lininterp(x, x0, x1, y0, y1):
  return (y0 + (x-x0)*(y1 - y0)/(x1 - x0))

def gordon_correction(stretch, n):
    cor = ( (stretch >= grdstretch[0] and stretch < grdstretch[1])*lininterp(stretch, grdstretch[0],grdstretch[1], grdstress[0],grdstress[1]) 
        +   (stretch >= grdstretch[1] and stretch < grdstretch[2])*lininterp(stretch, grdstretch[1],grdstretch[2], grdstress[1],grdstress[2])
        +   (stretch >= grdstretch[2] and stretch < grdstretch[3])*lininterp(stretch, grdstretch[2],grdstretch[3], grdstress[2],grdstress[3])
        +   (stretch >= grdstretch[3] and stretch < grdstretch[4])*lininterp(stretch, grdstretch[3],grdstretch[4], grdstress[3],grdstress[4]))
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

n = sn.Functional('n', [x,t,a,stretch,stretch_prev], 8*[400], 'tanh')    
L1 = (diff(n, t) + (stretch-stretch_prev)*(L0/dt) * diff(n, x) - gordon_correction(stretch,n)*f(x,a) + n*g(x))*(1+sign(n)) * 0.5 
#L1 = (diff(n, t) + (stretch-stretch_prev)*(L0/dt) * diff(n, x) - (1-n)*f(x,a) + n*g(x))* (1+sign(n)) * 0.5 
I1 = (t < TOL )*n
I2 = (1-sign(n))*n
D1 = sn.Data(n)

model = sn.SciModel([x,t,a,stretch,stretch_prev], [L1*LFACTOR, I1, I2, D1])  #load_weights_from='../models/isom-best_model-best.hdf5'


df = pd.read_csv('../data/pinn_data1.csv')
x_train = np.array(df['x'])
t_train = np.array(df['t'])
a_train = np.array(df['activation'])
stretch_train = np.array(df['stretch'])
stretch_prev_train = np.array(df['stretch_prev'])
n_train = np.array(df['n'])  

nzeros = 10000
t_train = np.append(t_train, np.zeros(nzeros))
n_train = np.append(n_train, np.zeros(nzeros))

x_train = np.append(x_train, np.random.choice(x_train, size=nzeros, replace=False))
a_train = np.append(a_train, np.zeros(nzeros))
stretch_train = np.append(stretch_train, np.ones(nzeros))
stretch_prev_train = np.append(stretch_prev_train, np.ones(nzeros))


h = model.train([x_train, t_train, a_train, stretch_train, stretch_prev_train], ['zeros','zeros', 'zeros', n_train], learning_rate=1e-4, batch_size=4096, epochs=15000, verbose=2, save_weights = {'path':'../models/best_model', 'best':True,'freq':1}, log_loss_gradients={'path':'../results/logs','freq':2000}, adaptive_weights={'method':'NTK', 'freq':100})
