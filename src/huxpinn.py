import numpy as np
import sys 
import pandas as pd 
import tensorflow as tf
import tensorflow_probability as tfp
import sciann as sn
from keras import backend as K
from sciann.utils.math import diff, sign
from numpy.random import seed
import random

''' to do: iskljuci svu stohastiku? '''
_seed = 137
seed(_seed)
random.seed(_seed)
tf.random.set_seed(_seed)
tf.compat.v1.set_random_seed(_seed)
sn.set_random_seed(_seed)


''' fixed parameters ''' 
TOL = 1e-3
f1_0 = 43.3 
h = 15.6
g1 = 10.0
g2 = 209.0
fzah = 4.0
L0 = 1100.0
dt = 1e-3
grdstretch = [0.6, 0.8, 0.95, 1.0, 1.64, 5.0]
grdstress = [0.0, 0.782, 1.0, 1.0, 0.0, 0.0]
''' fixed parameters '''

def lininterp(x, x0, x1, y0, y1):
  return (y0 + (x-x0)*(y1 - y0)/(x1 - x0))

def gordon_correction(stretch, n):
    cor = (stretch >= grdstretch[0] and stretch < grdstretch[1])*lininterp(stretch, grdstretch[0],grdstretch[1], grdstress[0],grdstress[1]) 
    cor+= (stretch >= grdstretch[1] and stretch < grdstretch[2])*lininterp(stretch, grdstretch[1],grdstretch[2], grdstress[1],grdstress[2])
    cor+= (stretch >= grdstretch[2] and stretch < grdstretch[3])*lininterp(stretch, grdstretch[2],grdstretch[3], grdstress[2],grdstress[3])
    cor+= (stretch >= grdstretch[3] and stretch < grdstretch[4])*lininterp(stretch, grdstretch[3],grdstretch[4], grdstress[3],grdstress[4])
    return (cor > n)*(cor - n)

def f(x,a):
    return (x >= -TOL and x <= (h+TOL))*(f1_0*a*x/h)

def g(x):
    return ((x < TOL)*g2 + 
            (x >= TOL and x <= (h+TOL))*(g1*x/h) + 
            (x > (h+TOL))*(fzah*g1*x/h))
      

x = sn.Variable('x')
t = sn.Variable('t')

#a = sn.Variable('a')
#stretch = sn.Variable('stretch')
#stretch_prev = sn.Variable('stretch_prev')
stretch = 1.0
stretch_prev = 1.0
a = 1.0

n = sn.Functional('n', [x,t], 8*[20], 'tanh')    
L1 = diff(n, t) - (1.0-n)*f(x,a) + n*g(x)
#L1_cor = diff(n, t) + (stretch-stretch_prev)*(L0/dt) * diff(n, x) - gordon_correction(stretch,n)*f(x,a) + n*g(x)
I1 = (t <= TOL )*n

model = sn.SciModel([t,x], [L1, I1]) 

t_data = np.linspace(0, 0.5, 100)
x_data = np.linspace(-21.0, 63.0, 200)
t_data, x_data = np.meshgrid( t_data, x_data )

h = model.train([t_data, x_data], 2*['zero'], learning_rate=1e-4, batch_size=512, epochs=10000,
                 stop_loss_value=1e-9, adaptive_weights={'method':'NTK', 'freq':500})
model.save_weights('../models/model.hdf5')

t_test = np.array([0,0.001,0.002, 0.4])
x_test = np.arange(-20.8,63,5.2)
#a_test = np.array([1.0])
#stretch_test = np.array([1.0])
#stretch_prev_test = np.array([1.0])
test_sample = np.meshgrid(t_test, x_test) 
prediction = n.eval(model,test_sample)

original_stdout = sys.stdout 
with open('../results/test.csv', 'w') as f:
  sys.stdout = f
  print('t,x,n')
  for tind in range(len(t_test)):
    for xind in range(len(x_test)):
      print(t_test[tind],',',x_test[xind],',',prediction[xind][tind])
  sys.stdout = original_stdout
    
