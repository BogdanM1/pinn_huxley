import numpy as np
import pandas as pd 
import sciann as sn
from sciann.utils.math import diff, sign

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
    cor = (stretch >= grdstretch[0] and stretch < grdstretch[1])*lininterp(stretch, grdstretch[0],grdstretch[1], grdstress[0],grdstress[1]) 
    cor+= (stretch >= grdstretch[1] and stretch < grdstretch[2])*lininterp(stretch, grdstretch[1],grdstretch[2], grdstress[1],grdstress[2])
    cor+= (stretch >= grdstretch[2] and stretch < grdstretch[3])*lininterp(stretch, grdstretch[2],grdstretch[3], grdstress[2],grdstress[3])
    cor+= (stretch >= grdstretch[3] and stretch < grdstretch[4])*lininterp(stretch, grdstretch[3],grdstretch[4], grdstress[3],grdstress[4])
    return (cor > n)*(cor - n)

def f(x,a):
    return (x >= .0 and x <= h)*(f1_0*a*x/h)

def g(x):
    return ((x < .0)*g2 + 
            (x >= .0 and x <= h)*(g1*x/h) + 
            (x > h)*(fzah*g1*x/h))
                       
      

x = sn.Variable('x')
t = sn.Variable('t')

n = sn.Functional('n', [x,t], 8*[20], 'tanh')    
L1 = (diff(n, t)  - (1.0-n)*f(x,1.0) + n*g(x) ) * (1+sign(n)) * 0.5
I1 = (t < TOL )*n
I2 = (1-sign(n))*n

model = sn.SciModel([x,t], [L1, I1, I2]) 


x_train = np.arange(-20.8,63,2.6)
t_train = np.linspace(0, 1.0, 500)

x_train, t_train = np.meshgrid( x_train, t_train )

h = model.train([x_train, t_train], ['zeros','zeros', 'zeros'], learning_rate=1e-4, batch_size=512, epochs=15000,
                 stop_loss_value=1e-9, adaptive_weights={'method':'NTK', 'freq':500}, verbose=2, save_weights = {'path':'../models/isom-best_model', 'best':True,'freq':1})
                 
model.save_weights('../models/isom-model.hdf5')



    
