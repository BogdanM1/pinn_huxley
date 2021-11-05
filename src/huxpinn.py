import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin
import pandas as pd 
from numpy.random import seed

_seed = 137
seed(_seed)

# fixed parameters:
TOL = 1e-9
f1_0 = 43.3 
h = 15.6
g1 = 10.0
g2 = 209.0
fzah = 4.0
L0 = 1100.0
dt = 1e-3
gordon_stretch = [0.6, 0.8, 0.95, 1.0, 1.64, 5.0]
gordon_stress = [0.0, 0.782, 1.0, 1.0, 0.0, 0.0]
#

def get_gordon(input_stretch):
    index = 0
    glen = len(gordon_stretch) 
    for igord  in range(glen):
        if(input_stretch <= gordon_stretch[igord]):
            index = igord
            break
    if(index==0): return gordon_stress[index]
    slope = (gordon_stress[index] - gordon_stress[index-1])
    slope /= (gordon_stretch[index] - gordon_stretch[index-1])
    return (gordon_stress[index - 1] + slope*(input_stretch - gordon_stretch[index - 1] ))

def f(x,a):
    if(x < 0): return 0
    if(x <= h): return (f1_0*a*x/h)
    return 0;

def g(x):
    if(x<0): return g2 
    if(x <=h): return (g1*x/h)
    return (fzah*g1*x/h)
    
def gordon_correction(input_stretch, nvalue):
    gord_value = get_gordon(input_stretch)
    if(gord_value > nvalue): return (gord_value-n)
    return 0.0     

x = sn.Variable('x')
t = sn.Variable('t')
#a = sn.Variable('a')
#stretch = sn.Variable('stretch')
#stretch_prev = sn.Variable('stretch_prev')

stretch = 1.0
stretch_prev = 1.0 
a = 1.0

n = sn.Functional('n', [t,x], 8*[100], 'sigmoid')
L1 = diff(n, t) - (stretch - stretch_prev)*(L0/dt) * diff(n, x) - (1-n)*f(x,a) + n*g(x)
L1_cor = diff(n, t) - (stretch - stretch_prev)*(L0/dt) * diff(n, x) - gordon_correction(stretch,n)*f(x,a) + n*g(x)
I1 = (1-sign(t - TOL)) *n

model = sn.SciModel([t,x], [L1, I1])
t_data, x_data = np.meshgrid( np.linspace(0, 2.0, 1000), np.linspace(-21.0, 63.0, 50) )

h = model.train([t_data, x_data], 2*['zero'], learning_rate=0.001, batch_size=1024, epochs=2000)
model.save_weights('../models/model.hdf5')

print('t,x,n')
for tval in [0, 0.001, 0.002]:
  for xval in np.arange(-20.8,62.4,5.2):
    prediction = n.eval(model,[np.array([tval]), np.array([xval])])
    print(tval,',',xval,',',prediction[0])