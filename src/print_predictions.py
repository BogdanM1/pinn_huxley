import sys

def print_predictions(prediction, x_test, t_test, filepath):
    original_stdout = sys.stdout 
    with open(filepath, 'w') as f:
      sys.stdout = f
      print('t,x,n')
      for tind in range(len(t_test)):
        for xind in range(len(x_test)):
          print(t_test[tind],',',x_test[xind],',',prediction[tind][xind][0][0][0])
      sys.stdout = original_stdout