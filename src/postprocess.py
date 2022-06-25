import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error
import math
import os


num_tests = 165
writeSimulationResults = True
results_dir = '../results/'
data_noiter = pd.read_csv("../data/dataMexieNoIter.csv")
target_columns  = [7, 8]

def print_metrics(sig_orig, dsig_orig, sig_pred, dsig_pred):
		mean_sig_orig = np.mean(sig_orig)
		mean_dsig_orig = np.mean(dsig_orig)
		mean_sig_pred = np.mean(sig_pred)
		mean_dsig_pred = np.mean(dsig_pred)
		mean_sig_orig_diff = np.array([x - mean_sig_orig for x in sig_orig])
		mean_dsig_orig_diff = np.array([x - mean_dsig_orig for x in dsig_orig])
		mean_sig_pred_diff = np.array([x - mean_sig_pred for x in sig_pred])
		mean_dsig_pred_diff = np.array([x - mean_dsig_pred for x in dsig_pred])
       
		rmse_sig = math.sqrt(mean_squared_error(sig_orig, sig_pred ))
		rmse_dsig = math.sqrt(mean_squared_error(dsig_orig,  dsig_pred))
		max_sig = max(abs(sig_orig - sig_pred))
		max_dsig = max(abs(dsig_orig - dsig_pred))   
		min_sig = min(abs(sig_orig - sig_pred))
		min_dsig = min(abs(dsig_orig - dsig_pred))
		rse_sig = rmse_sig/math.sqrt(sum(mean_sig_orig_diff*mean_sig_orig_diff))
		rse_dsig = rmse_dsig/math.sqrt(sum(mean_dsig_orig_diff*mean_dsig_orig_diff))
		corr_sig = (sum(mean_sig_orig_diff*mean_sig_pred_diff))
		corr_sig = corr_sig/math.sqrt(sum(mean_sig_orig_diff*mean_sig_orig_diff)*sum(mean_sig_pred_diff*mean_sig_pred_diff))
		corr_dsig = (sum(mean_dsig_orig_diff*mean_dsig_pred_diff))
		corr_dsig = corr_dsig/math.sqrt(sum(mean_dsig_orig_diff*mean_dsig_orig_diff)*sum(mean_dsig_pred_diff*mean_dsig_pred_diff))    
		print(str(rmse_sig)+','+str(rmse_dsig)+','+str(max_sig)+','+str(max_dsig)+','+str(min_sig)+','+str(min_dsig)
    +','+str(rse_sig)+','+str(rse_dsig)+','+str(corr_sig)+','+str(corr_dsig))

def drawGraphRes(x, y1, y2, name1, name2, title, testid, dotted=True):
    global results_dir
    plt.figure(figsize=(5, 4), dpi=300)
    if(dotted):
      plt.plot(x, y1, marker='o', markersize=1, color='indigo', linestyle='None') 
      plt.plot(x, y2,  marker='o', markersize=1,  color='#F092DA', linestyle='None')      
    else: 
      plt.plot(x, y1, linewidth=2.0, color='indigo', linestyle='-')   
      plt.plot(x, y2, linewidth=2.0, color='#F092DA', linestyle='--') 
    plt.xlabel('Time $[s]$')
    plt.xlim(left=0)
    plt.ylabel(title + ' $[pN/nm^2]$')
    plt.ylim(bottom=0, top=1.2*plt.ylim()[1])
    plt.title('Example ' + str(testid) + ' - ' + title, loc = 'left')
    plt.legend([name1, name2], loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(results_dir + title + str(testid) + '.png')
    plt.close()

def list_to_num(numList):         
    s = map(str, numList)   
    s = ''.join(s)          
    s = int(s)              
    return s
	
def drawTestResults():
    global results_dir
    for file_name in os.listdir(results_dir):
        if not file_name.startswith('data') and not file_name.startswith('simulation'):
            continue
        data = pd.read_csv(results_dir + file_name)
        time = np.array(data['time'])
        sigma = np.array(data['sigma'])
        delta_sigma = np.array(data['delta_sigma'])
        sigma_pred = np.array(data['sigma pred'])
        delta_sigma_pred = np.array(data['delta_sigma pred'])
        testid = list_to_num([int(s) for s in file_name if s.isdigit()])
        if file_name.startswith('data'):
            drawGraphRes(time, sigma, sigma_pred, 'Original model', 'Surrogate model', 'Stress', testid)
            drawGraphRes(time, delta_sigma, delta_sigma_pred, 'Original model', 'Surrogate model','Stress derivative', testid, dotted=True)
        else:
            drawGraphRes(time, sigma, sigma_pred, 'Original model', 'Surrogate model', 'Stress (simulation)', testid)
            #drawGraphRes(time, delta_sigma, delta_sigma_pred, 'Original model', 'Surrogate model', 'Stress derivative (simulation)', testid)
            

if(writeSimulationResults):
	print('simulation')
	print('rmse(stress), rmse(stress derivative), max_err(stress), max_err(stress derviative), min_err(stress), min_err(stress derivative), rse(stress), rse(stress derviative), corr(stress), corr(stress derivative)') 
	for i in range(0,num_tests):
	    try:
	        indices       = data_noiter.index[data_noiter['testid'] == (i+1)].tolist()
	        original_data = np.array(data_noiter)[indices, :]
	        prediction = pd.read_csv(results_dir + "surroHuxley"+str(i+1)+".csv", sep='\s*,\s*', engine='python')
	        prediction = np.array(prediction.loc[::4, ['sigma','delta_sigma']])
	        original_data = original_data[:len(prediction),]
	        print_metrics(original_data[:,target_columns[0]], original_data[:,target_columns[1]], prediction[:, 0], prediction[:, 1])
        	df = pd.DataFrame(data = { 'time': original_data[:,0],
                                     'sigma': original_data[:,target_columns[0]],
                                     'delta_sigma': original_data[:,target_columns[1]],
                                     'sigma pred': prediction[:, 0],
                                     'delta_sigma pred': prediction[:,1]})
	        df.to_csv(results_dir + 'simulation_pred_test' + str(i+1) + '.csv', index=False)
	    except:
	        print("Error during processing test No. " + str(i+1))

drawTestResults()
'''
for file_name in os.listdir(results_dir):
    if(file_name.endswith('.csv')):
        os.unlink(results_dir + file_name)
'''
