import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin
import pandas as pd 
from numpy.random import seed

_seed = 137
seed(_seed)

TOL = 1e-9
# fixed parameters:
f1_0 = 43.3 
h = 15.6
g1 = 10.0
g2 = 209.0
fzah = 4.0
#

def f(x,a):
    if(x < 0): return 0
    if(x <= h): return (f1_0*a*x/h)
    return 0;

def g(x):
    if(x<0): return g2 
    if(x <=h): return (g1*x/h)
    return (fzah*g1*x/h)

x = sn.Variable('x')
t = sn.Variable('t')
a = sn.Variable('a')
v = sn.Variable('v')

n = sn.Functional('n', [t,x,a,v], 8*[20], 'tanh')
L1 = diff(n, t) - v * diff(n, x) - (1-n)*f(x,a) + n*g(x)
I1 = (1-sign(t - TOL)) *n

model = sn.SciModel([t,x,a,v], [L1, I1])
x_data, t_data, a_data, v_data = np.meshgrid( np.linspace(-21.0, 63.0, 100), 
                                              np.linspace(0, 2.0, 100), 
                                              np.linspace(1, 1, 1), 
                                              np.linspace(1, 1, 1))

h = model.train([x_data, t_data, a_data, v_data], 2*['zero'], learning_rate=0.001, batch_size=1024, epochs=20000)
model.save_weights('../models/model.hdf5')

