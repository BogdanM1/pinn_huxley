import numpy as np
import random  
import pandas as pd 
import sciann as sn
from sciann.utils.math import diff, sign
from print_predictions import print_predictions

''' fixed parameters ''' 
TOL = 1e-2
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

n = sn.Functional('n', [x,t], 8*[20], 'tanh')    
L1 = (diff(n, t) + (stretch-stretch_prev)*(L0/dt) * diff(n, x) - gordon_correction(stretch,n)*f(x,a) + n*g(x))* (1+sign(n)) * 0.5
I1 = (t < TOL )*n
I2 = (1-sign(n))*n

model = sn.SciModel([x,t,a,stretch,stretch_prev], [L1, I1, I2]) 

nsamples = 1000000
nzeros = 10000
df = pd.DataFrame()
df['x'] = np.random.choice(np.arange(-20.8,63,0.13), nsamples) 
df['t'] = np.append(np.random.choice(np.linspace(0, 2.0, 1000), nsamples-nzeros), np.zeros(nzeros))
df['a'] = np.random.choice(np.linspace(0.0, 1.0, 1000), nsamples) 
df['stretch'] = np.random.choice(np.linspace(1.25, .75, 1000), nsamples)
df['stretch_prev'] = df['stretch'] + [random.uniform(-0.1,0.1) for i in range(nsamples)] 
df = df.drop_duplicates()
df.to_csv("../data/input_data.csv", index = False)

x_train = np.array(df['x']) 
t_train = np.array(df['t']) 
a_train = np.array(df['a'])
stretch_train = np.array(df['stretch'])
stretch_prev_train =  np.array(df['stretch_prev'])  

h = model.train([x_train, t_train, a_train, stretch_train, stretch_prev_train], 3*['zero'], learning_rate=1e-4, batch_size=512, epochs=10000,
                 stop_loss_value=1e-9, adaptive_weights={'method':'NTK', 'freq':500})
model.save_weights('../models/model.hdf5')


x_test = np.arange(-20.8,63,5.2)
t_test = np.array([0,0.001,0.002, 0.4])
a_test = np.array([1.0])
stretch_test = np.array([1.0])
stretch_prev_test = np.array([1.0])
test_sample = np.meshgrid(x_test, t_test, a_test, stretch_test, stretch_prev_test) 
prediction = n.eval(model,test_sample)
print_predictions(prediction, x_test, t_test, '../results/test.csv')

    
