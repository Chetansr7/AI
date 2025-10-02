# dataset
from sklearn import datasets
import pandas as pd
diabetes = datasets.load_diabetes()
diabetes = {
  'attributes': pd.DataFrame(diabetes.data, columns=diabetes.feature_names),
  'target': pd.DataFrame(diabetes.target, columns=['diseaseProgression'])
}

# split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(diabetes['attributes'], diabetes['target'], test_size=0.2, random_state=1)
diabetes['train'] = {
  'attributes': x_train,
  'target': y_train
}
diabetes['test'] = {
  'attributes': x_test,
  'target': y_test
}

# regression
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(5)

input_columns = ['age', 'bmi']
x_train = diabetes['train']['attributes'][input_columns]
y_train = diabetes['train']['target'].diseaseProgression
knr.fit(x_train, y_train)

x_test = diabetes['test']['attributes'][input_columns]
y_test = diabetes['test']['target'].diseaseProgression
y_predict = knr.predict(x_test)

print(pd.DataFrame(list(zip(y_test,y_predict)), columns=['target', 'predicted']))
print(f'Accuracy: {knr.score(x_test,y_test):.4f}')

# visualisation
import matplotlib.pyplot as plt
from matplotlib import cm

# colormap
dia_cm = cm.get_cmap('Reds')

# calculate decision boundaries
import numpy as np
x_min = diabetes['attributes'][input_columns[0]].min()
x_max = diabetes['attributes'][input_columns[0]].max()
x_range = x_max - x_min
x_min = x_min - 0.1*x_range
x_max = x_max + 0.1*x_range
y_min = diabetes['attributes'][input_columns[1]].min()
y_max = diabetes['attributes'][input_columns[1]].max()
y_range = y_max - y_min
y_min = y_min - 0.1*y_range
y_max = y_max + 0.1*y_range
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), np.arange(y_min, y_max, .01*y_range))
z = knr.predict(list(zip(xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)

## display decision boundary
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=dia_cm)

## plot training and testing data
plt.scatter(x_train[input_columns[0]], x_train[input_columns[1]],
            c=y_train, label='Training data', cmap=dia_cm, 
            edgecolor='black', linewidth=1, s=150)            
plt.scatter(x_test[input_columns[0]], x_test[input_columns[1]],
            c=y_test, marker='*', label='Testing data', cmap=dia_cm,
            edgecolor='black', linewidth=1, s=150)

## label the graph
plt.xlabel(input_columns[0])
plt.ylabel(input_columns[1])
plt.legend()
plt.colorbar()

## show the graph
# plt.show()

# loop to compare accuracy of prediction at different value of k
k_list = []
accuracy_list = []
for k in range(1, len(x_train)+1):
  k_list.append(k)
  knr = KNeighborsRegressor(k)
  knr.fit(x_train, y_train)
  accuracy_list.append(knr.score(x_test, y_test))
plt.figure()
plt.scatter(k_list, accuracy_list)
plt.xlabel('$k$')
plt.ylabel('Accuracy')
plt.title('Comparison of accuracy for different k')
plt.show()