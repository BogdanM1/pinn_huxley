import deepxde as dde
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras import backend as K
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense

dde.config.set_random_seed(100)
os.system("rm ../models/*")

''' fixed parameters ''' 
f1_0 = 43.3 
h = 15.6
g1 = 10.0
g2 = 208.0
fzah = 1.0
L0 = 1100.0
dt = 1e-3
grdstretch = [0.6, 0.8, 0.95, 1.0, 1.64, 5.0]
grdstress = [0.0, 0.782, 1.0, 1.0, 0.0, 0.0]


def lininterp(x, x0, x1, y0, y1):
  return (y0 + (x-x0)*(y1 - y0)/(x1 - x0))

def gordon_correction(stretch, n):
    cor = ( 0.25*(1 + tf.sign(stretch - grdstretch[0])) * ( 1 - tf.sign(stretch - grdstretch[1]))*lininterp(stretch, grdstretch[0],grdstretch[1], grdstress[0],grdstress[1]) 
        +   0.25*(1 + tf.sign(stretch - grdstretch[1])) * ( 1 - tf.sign(stretch - grdstretch[2]))*lininterp(stretch, grdstretch[1],grdstretch[2], grdstress[1],grdstress[2])
        +   0.25*(1 + tf.sign(stretch - grdstretch[2])) * ( 1 - tf.sign(stretch - grdstretch[3]))*lininterp(stretch, grdstretch[2],grdstretch[3], grdstress[2],grdstress[3])
        +   0.25*(1 + tf.sign(stretch - grdstretch[3])) * ( 1 - tf.sign(stretch - grdstretch[4]))*lininterp(stretch, grdstretch[3],grdstretch[4], grdstress[3],grdstress[4])
        +   0.25*(1 + tf.sign(stretch - grdstretch[4])) * ( 1 - tf.sign(stretch - grdstretch[5]))*lininterp(stretch, grdstretch[4],grdstretch[5], grdstress[4],grdstress[5]))
    return 0.5*(1 + tf.sign(cor - n))*(cor - n)


def f(x,a):
    return (1+tf.sign(x)) * (1-tf.sign(x-h)) * (f1_0*a*x/h) * 0.25


def g(x):
    return (0.5 * (1-tf.sign(x)) * g2 + 
           0.25 * (1+tf.sign(x)) * (1-tf.sign(x-h)) * (g1*x/h) + 
           0.5 * (1+tf.sign(x-h)) * (fzah*g1*x/h))

# n = n(x,t,a,l,lp)
def pde(xx, n):
    dn_dt = dde.grad.jacobian(n, xx, i=0, j=1)
    dn_dx = dde.grad.jacobian(n, xx, i=0, j=0)
    # loss = dn_dt + L0/dt*(xx[:,3:4] - xx[:,5:6])*dn_dx - (1.0-n) * f(xx[:,0:1], xx[:,2:3]) + n*g(xx[:,0:1])
    #loss = dn_dt + L0/dt*(xx[:,3:4] - xx[:,5:6])*dn_dx - gordon_correction(xx[:,3:4],n) * f(xx[:,0:1], xx[:,2:3]) + n*g(xx[:,0:1])
    loss = dn_dt  - (1.0-n) * f(xx[:,0:1], 1.0) + n*g(xx[:,0:1])
    return loss + n*(1-tf.sign(n))
    
def boundary(x, _):
    return dde.utils.isclose(x[1], 0) 
 
def func(x):
    return 0.0  
  
geom = dde.geometry.geometry_nd.Hypercube([-2.08, .0],[63., .4])
ic1 = dde.icbc.DirichletBC(geom, func, boundary)
data = dde.data.PDE(geom, pde, [ic1], num_domain=50000, num_boundary=1000)
net = dde.nn.FNN([2] + [40] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss_weights=[1.e-1, 1])
model.train(150000)
model.compile("L-BFGS", loss_weights=[1.e-1, 1])
losshistory, train_state = model.train()
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)

model.save("../models/tmpmodel")
#print(model.predict(np.array([[14, .1, 1., 1., .4]])))
os.system("python3 convert_ckpt_to_pb.py "+ str(train_state.step))
os.system("python3 test_pb_file.py")

