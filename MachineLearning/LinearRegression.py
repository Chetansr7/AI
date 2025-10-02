# Import from sklearn libraru
from sklearn import datasets
diabetes = datasets.load_diabetes()

# Import pandas library
import pandas as pd

# Import function to split the train-test data given a percentage
from sklearn.model_selection import train_test_split

# Import matplotlib library
import matplotlib.pyplot as plt

import math
import sys


# Print description of diabetes dataset
print(diabetes.DESCR)

# Convert dataset into pandas DataFrame
dt = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y= pd.DataFrame(diabetes.target, columns=['target'])

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(dt, y, test_size=0.2)


# Define model of a regression line
def model(x, m, c):
    y = (m*x) + c
    return y

# Define cost function
def cost(y,yh):
    J = ((y-yh)**2).mean()
    return J

# Define derivatives of the cost
def derivatives(x,y,yh):
    return {
        'm': ((y-yh)*x).mean()*-2,
        'c': (y-yh).mean()*-2
        }

# Define initial values
learningrate = 0.1
m = []
c= []
J = []
m.append(0)
c.append(0)
J.append(cost(y_train['target'], X_train['bmi'].apply(lambda x: model(x, m [-1], c[-1]))))

# Define termination conditions
J_min = 0.01
del_J_min = 0.0001
max_iter = 10000

def getdelJ():
    if len(J) > 1:
        return math.fabs(J[-1] - J[-2]/J[-1])
    else:
        return inf
    

# Main Loop
while J[-1] > J_min and getdelJ() > del_J_min and len(J) < max_iter:
    der = derivatives(X_train['bmi']. y_train['target'], X_train['bmi'].append)
    m.append(m[-1] - learningrate * der['m'])
    c.append(c[-1] - learningrate * der['c'])
    J.append(cost(y_train['target'], X_train['bmi'].apply(lambda x: model(x, m, c))))
    
    print('.', end='')
    sys.stdout.flush()
    
    if line:
        line[0].remove()
    line = plt.plot(X_train['bmi'], X_train['bmi'].apply(lambda x: model(x, m[-1], c[-1])), '-', color='green')
    plt.pause(0.001)
    
    
y_train_pred = X_train['bmi'].apply(lambda x: model(x, m[-1], c[-1]))
y_test_pred = X_test['bmi'].apply(lambda x: model(x, m[-1], c[-1]))
print('\nAlgorithm terminated with')
print(f'  {len(J)} iterations,')
print(f'  m {m[-1]}')
print(f'  c {c[-1]}')
print(f'  training cost {J[-1]}')
testcost = cost(y_test['target'], y_test_pred)
print(f'  testing cost {testcost}')


plt.figure()
plt.scatter(X_test['bmi'], y_test['target'], color='red')
plt.plot(X_test['bmi'], \
         X_test['bmi'].apply(lambda x: model(x, m[-1], c[-1])), \
         '-', color='green')
plt.title('Testing data')


