import os
import random
import numpy as np
import pandas as pd 
import sciann as sn
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from skopt import gp_minimize
from sciann.utils.math import diff, sign
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import  Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

# ... Optimization options ...
optimize_both_objectives = True
use_labels = False
nfeatures = 2
n_iterations = 100
nepochs = 100
nntk = 30  
nlaymin = 1
nneuronsmin = 10
nlaymax = 10
nneuronsmax = 100

''' fixed Huxley parameters ''' 
TOL = 1e-3
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

# paths to save the plots
evaluations_plot_path = "../results/evaluations_plot.png"
objective1_plot_path = "../results/objective1_plot.png"
objective2_plot_path = "../results/objective2_plot.png"
convergence_plot_path = "../results/convergence_plot.png"

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

df = pd.read_csv('../data/input_iso.csv')
df = df.drop_duplicates()
x_train = np.array(df['x'])
t_train = np.array(df['t'])
a_train = np.array(df['a'])
stretch_train = np.array(df['stretch'])
stretch_prev_train = np.array(df['stretch_prev'])
n_train = np.array(df['n'])  
v_train = (stretch_train-stretch_prev_train)*(L0/dt)

# points satisfying inital condition 
nzeros = (int)(len(n_train)/10)
t_train = np.append(t_train, np.zeros(nzeros))
n_train = np.append(n_train, np.zeros(nzeros))
x_train = np.append(x_train, np.random.choice(x_train, size=nzeros, replace=False))
a_train = np.append(a_train,  np.zeros(nzeros))
stretch_train = np.append(stretch_train, np.ones(nzeros))
stretch_prev_train = np.append(stretch_prev_train, np.ones(nzeros))
v_train = np.append(v_train, np.zeros(nzeros))

if(use_labels):
  target = ['zeros', 'zeros', 'zeros', n_train]
else:
  target = ['zeros', 'zeros', 'zeros'] 

# Define the multi-objective objective function
def objective(params):
    nlayers = int(params[0])
    nneurons = int(params[1])
    activation = params[2]
    learning_rate = params[3]
  
    if(nfeatures==2):
      features = [x,t]
    else: 
      features = [x,t,a,stretch, stretch_prev]

    n = sn.Functional('n', features, nlayers*[nneurons], activation)    

    L1 = (diff(n, t) + (stretch-stretch_prev)*(L0/dt) * diff(n, x) - (1-n)*f(x,a) + n*g(x))* (1+sign(n)) * 0.5 
    I1 = (t < TOL )*n
    I2 = (1-sign(n))*n
    D1 = sn.Data(n) 
    
    if(use_labels):
      losses = [L1, I1, I2, D1]  
    else:
      losses = [L1, I1, I2]

    model = sn.SciModel([x,t,a,stretch,stretch_prev], losses)


    # Train the model with the specified learning rate and obtain the final loss
    history = model.train([x_train, t_train, a_train, stretch_train, stretch_prev_train], target, 
                          learning_rate=learning_rate, batch_size=512, epochs=nepochs, verbose=2,
                          adaptive_weights={'method': 'NTK', 'freq': nntk})

    final_loss = history.history['loss'][-1]
    
    # Calculate the size of the neural network
    nn_size = nlayers * nneurons
    if(optimize_both_objectives):
      return final_loss*100.0 + nn_size/(nlaymax*nneuronsmax)
    return final_loss

# Create the optimizer
dimensions = [(nlaymin, nlaymax),     # Number of layers
              (nneuronsmin, nneuronsmax),   # Number of neurons
              ('relu', 'tanh', 'sigmoid','selu'),  # Activation function
              (1e-6, 1e-2, 'log-uniform')]  # Learning rate


results = gp_minimize(func=objective,
                      dimensions=dimensions,
                      n_calls=n_iterations,
                      n_jobs=-1,  # Use all available cores for parallelization
                      verbose=True)

# Get the best hyperparameters and the corresponding losses
best_params = results.x
best_losses = results.fun
print(f"Best Losses: {best_losses}, Best Params: {best_params}")                      

# Plot the evaluated points and their losses
plt.figure(figsize=(10, 6))
plot_evaluations(results)
plt.savefig(evaluations_plot_path)
plt.close()

# Plot the convergence
plt.figure(figsize=(10, 6))
plot_convergence(results)
plt.title("Convergence Plot")
plt.xlabel("Iteration")
plt.ylabel("Best Loss")
plt.savefig(convergence_plot_path)
plt.close()

# Plot the estimated objective function for the first objective
plt.figure(figsize=(10, 6))
plot_objective(results)
plt.savefig(objective1_plot_path)
plt.close()


