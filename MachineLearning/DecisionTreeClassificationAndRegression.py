# dataset
# Prepare dataset for classification
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
iris = {
  'attributes': pd.DataFrame(iris.data, columns=iris.feature_names),
  'target': pd.DataFrame(iris.target, columns=['species']),
  'targetNames': iris.target_names
}

# split
from sklearn.model_selection import train_test_split

# Split dataset into 80-20 proportion
x_train, x_test, y_train, y_test = train_test_split(iris['attributes'], iris['target'], test_size=0.2, random_state=1)

iris['train'] = {
  'attributes': x_train,
  'target': y_train
}
iris['test'] = {
  'attributes': x_test,
  'target': y_test
}

# Import the class for decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Instantiate an object of DecisionTreeClassifier class with gini impurity as the split criterion
dtc = DecisionTreeClassifier(criterion='gini')

# Train the classifier with training data
dtc.fit(iris['train']['attributes'], iris['train']['target'])

# .predict function is used ot predict the species of the testing data
predicts = dtc.predict(iris['test']['attributes'])

# Comparing the predicted value and the target value of the test data
print(pd.DataFrame(list(zip(iris['test']['target'].species,predicts)), columns=['target', 'predicted']))

# Calculate the accuracy of the predicted value
accuracy = dtc.score(iris['test']['attributes'],iris['test']['target'].species)
print(f'Accuracy: {accuracy:.4f}')

# Import the matplotlib.pyplot library and the function to visualise the tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualise the decision tree model
plt.figure(figsize=[10,10])
tree = plot_tree(dtc, feature_names=iris['attributes'].columns.tolist(), 
                 class_names=iris['targetNames'], filled=True, rounded=True)

max_depth = [1,2,3,5,6,7,20]
training_accuracy = []
testing_accuracy = []

for md in max_depth:
    dtc = DecisionTreeClassifier(max_depth=md)
    dtc.fit(iris['train']['attributes'], iris['train']['target'].species)
    train = dtc.score(iris['train']['attributes'], iris['train']['target'].species)
    test = dtc.score(iris['test']['attributes'], iris['test']['target'].species)
    training_accuracy.append(train)
    testing_accuracy.append(test)
    
plt.figure()
plt.scatter(max_depth, training_accuracy, label = 'training accuracy')
plt.scatter(max_depth, testing_accuracy, label = 'testing accuracy')
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.legend()

plt.show()
    
## Visualisation of decision surface ##

# Instantiate the classifier without defining the maximum depth and train the model
dtc = DecisionTreeClassifier()
input_cols = iris['train']['attributes'].columns[:2].tolist()
dtc.fit(iris['train']['attributes'][input_cols], iris['train']['target'].species)

plt.figure(figsize=[50,50])
plot_tree(dtc, feature_names=input_cols, 
          class_names=iris['targetNames'], filled=True, rounded=True)
plt.savefig('classificationDecisionTreeWithNoMaxDepth.png')

from matplotlib import cm
from matplotlib.colors import ListedColormap
colormap = cm.get_cmap('tab20')
cm_dark = ListedColormap(colormap.colors[::2])
cm_light = ListedColormap(colormap.colors[1::2])

import numpy as np
x_min = iris['attributes'][input_cols[0]].min()
x_max = iris['attributes'][input_cols[0]].max()
x_range = x_max - x_min
x_min = x_min - 0.1 * x_range
x_max = x_max + 0.1 * x_range
y_min = iris['attributes'][input_cols[1]].min()
y_max = iris['attributes'][input_cols[1]].max()
y_range = y_max - y_min
y_min = y_min - 0.1 * y_range
y_max = y_max + 0.1 * y_range
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), 
                    np.arange(y_min, y_max, .01*y_range))
z = dtc.predict(list(zip(xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cm_light)

plt.scatter(iris['train']['attributes'][input_cols[0]],   
            iris['train']['attributes'][input_cols[1]], 
            c=iris['train']['target'].species, cmap=cm_dark, s=200,
            label='Training data', edgecolor='black', linewidth=1)
plt.scatter(iris['test']['attributes'][input_cols[0]], 
            iris['test']['attributes'][input_cols[1]], 
            c=iris['test']['target'].species, cmap=cm_dark, s=200,
            label='Testing data', edgecolor='black', linewidth=1,
            marker='*')
train_acc = dtc.score(iris['train']['attributes'][input_cols], 
                      iris['train']['target'].species)
test_acc = dtc.score(iris['test']['attributes'][input_cols], 
                    iris['test']['target'].species)
plt.title(f'training: {train_acc:.3f}, testing: {test_acc:.3f}')
plt.xlabel(input_cols[0])
plt.ylabel(input_cols[1])
plt.legend()

# Overfitting
dtc = DecisionTreeClassifier(max_depth = 3)
input_cols = iris['train']['attributes'].columns[:2].tolist()
dtc.fit(iris['train']['attributes'][input_cols], iris['train']['target'].species)

plt.figure(figsize=[50,50])
plot_tree(dtc, feature_names=input_cols, 
          class_names=iris['targetNames'], filled=True, rounded=True)
plt.savefig('classificationDecisionTreeWithMaxDepth3.png')

from matplotlib import cm
from matplotlib.colors import ListedColormap
colormap = cm.get_cmap('tab20')
cm_dark = ListedColormap(colormap.colors[::2])
cm_light = ListedColormap(colormap.colors[1::2])

import numpy as np
x_min = iris['attributes'][input_cols[0]].min()
x_max = iris['attributes'][input_cols[0]].max()
x_range = x_max - x_min
x_min = x_min - 0.1 * x_range
x_max = x_max + 0.1 * x_range
y_min = iris['attributes'][input_cols[1]].min()
y_max = iris['attributes'][input_cols[1]].max()
y_range = y_max - y_min
y_min = y_min - 0.1 * y_range
y_max = y_max + 0.1 * y_range
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), 
                    np.arange(y_min, y_max, .01*y_range))
z = dtc.predict(list(zip(xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cm_light)

plt.scatter(iris['train']['attributes'][input_cols[0]],   
            iris['train']['attributes'][input_cols[1]], 
            c=iris['train']['target'].species, cmap=cm_dark, s=200,
            label='Training data', edgecolor='black', linewidth=1)
plt.scatter(iris['test']['attributes'][input_cols[0]], 
            iris['test']['attributes'][input_cols[1]], 
            c=iris['test']['target'].species, cmap=cm_dark, s=200,
            label='Testing data', edgecolor='black', linewidth=1,
            marker='*')
train_acc = dtc.score(iris['train']['attributes'][input_cols], 
                      iris['train']['target'].species)
test_acc = dtc.score(iris['test']['attributes'][input_cols], 
                    iris['test']['target'].species)
plt.title(f'training: {train_acc:.3f}, testing: {test_acc:.3f}')
plt.xlabel(input_cols[0])
plt.ylabel(input_cols[1])
plt.legend()

plt.show()

# Import class for decision tree regressor
from sklearn.tree import DecisionTreeRegressor

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

# Instantiate an object of DecisionTreeRegressor class
dtr = DecisionTreeRegressor(max_depth=None)

# Train the classifier with training data
dtr.fit(diabetes['train']['attributes'], diabetes['train']['target'])

# .predict function is used to predict the disease progression of the testing data.
predicts = dtr.predict(diabetes['test']['attributes'])

# Comparing the predicted value and the target value of the test data.
print(pd.DataFrame(list(zip(diabetes['test']['target'].diseaseProgression,predicts)), 
                  columns=['target', 'predicted']))

# Calculate the accuracy of the predicted value.
accuracy = dtr.score(diabetes['test']['attributes'],
                    diabetes['test']['target'].diseaseProgression)
print(f'Accuracy: {accuracy:.4f}')

# Import the matplotlib.pyplot library and the function to visualise the tree.
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualise the decision tree model.
plt.figure(figsize=[10,10])
tree = plot_tree(dtr, feature_names=diabetes['attributes'].columns.tolist(), 
                filled=True, rounded=True)


max_depth = [1, 2, 3, 5, 7, 20]

training_accuracy = []
testing_accuracy = []

for md in max_depth:
    dtr = DecisionTreeRegressor(max_depth=md)
    dtr.fit(diabetes['train']['attributes'], diabetes['train']['target'].diseaseProgression)

    train_acc = dtr.score(diabetes['train']['attributes'], diabetes['train']['target'].diseaseProgression)
    test_acc = dtr.score(diabetes['test']['attributes'], diabetes['test']['target'].diseaseProgression)
    training_accuracy.append(train_acc)
    testing_accuracy.append(test_acc)

# Plotting the results
plt.figure()
plt.plot(max_depth, training_accuracy, label='Training Accuracy')
plt.plot(max_depth, testing_accuracy, label='Testing Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Regressor')

plt.show()

## Visualisation of decision surface ##

# Instantiate the classifier without defining the maximum depth and train the model.
dtr = DecisionTreeRegressor()
input_cols = ['age', 'bmi']
dtr.fit(diabetes['train']['attributes'][input_cols], 
        diabetes['train']['target'].diseaseProgression)

# Plot the decision tree
plt.figure(figsize=[50,50])
plot_tree(dtr, feature_names=input_cols, filled=True, rounded=True)
plt.savefig('regressionDecisionTreeWithNoMaxDepth.png')

# Prepare the colourmaps
from matplotlib import cm
dia_cm = cm.get_cmap('Reds')

# Create the decision surface
x_min = diabetes['attributes'][input_cols[0]].min()
x_max = diabetes['attributes'][input_cols[0]].max()
x_range = x_max - x_min
x_min = x_min - 0.1 * x_range
x_max = x_max + 0.1 * x_range
y_min = diabetes['attributes'][input_cols[1]].min()
y_max = diabetes['attributes'][input_cols[1]].max()
y_range = y_max - y_min
y_min = y_min - 0.1 * y_range
y_max = y_max + 0.1 * y_range
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), 
                    np.arange(y_min, y_max, .01*y_range))
z = dtr.predict(list(zip(xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)


# Plot the decision surface
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=dia_cm)

# Plot the training and testing data.
plt.scatter(diabetes['train']['attributes'][input_cols[0]],          
            diabetes['train']['attributes'][input_cols[1]], 
            c=diabetes['train']['target'].diseaseProgression, 
            label='Training data', cmap=dia_cm, 
            edgecolor='black', linewidth=1, s=150)
plt.scatter(diabetes['test']['attributes'][input_cols[0]],   
            diabetes['test']['attributes'][input_cols[1]], 
            c=diabetes['test']['target'].diseaseProgression, marker='*', 
            label='Testing data', cmap=dia_cm, 
            edgecolor='black', linewidth=1, s=150)
plt.xlabel(input_cols[0])
plt.ylabel(input_cols[1])
plt.legend()
plt.colorbar()

# Overfitting 
dtr = DecisionTreeRegressor(max_depth=3)
dtr.fit(diabetes['train']['attributes'][input_cols], 
        diabetes['train']['target'].diseaseProgression)

plt.figure(figsize=[50, 50])
plot_tree(dtr, feature_names=input_cols, filled=True, rounded=True)
plt.savefig('regressionDecisionTreeWithMaxDepth3.png')


x_min = diabetes['attributes'][input_cols[0]].min()
x_max = diabetes['attributes'][input_cols[0]].max()
x_range = x_max - x_min
x_min = x_min - 0.1 * x_range
x_max = x_max + 0.1 * x_range
y_min = diabetes['attributes'][input_cols[1]].min()
y_max = diabetes['attributes'][input_cols[1]].max()
y_range = y_max - y_min
y_min = y_min - 0.1 * y_range
y_max = y_max + 0.1 * y_range
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), 
                     np.arange(y_min, y_max, .01*y_range))
z = dtr.predict(list(zip(xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)


plt.figure()
plt.pcolormesh(xx, yy, z, cmap=dia_cm)

plt.scatter(diabetes['train']['attributes'][input_cols[0]], 
            diabetes['train']['attributes'][input_cols[1]], 
            c=diabetes['train']['target'].diseaseProgression, 
            cmap=dia_cm, edgecolor='black', linewidth=1, 
            s=150, label='Training data')
plt.scatter(diabetes['test']['attributes'][input_cols[0]], 
            diabetes['test']['attributes'][input_cols[1]], 
            c=diabetes['test']['target'].diseaseProgression, 
            cmap=dia_cm, marker='*', edgecolor='black', 
            linewidth=1, s=150, label='Testing data')
train_acc = dtr.score(diabetes['train']['attributes'][input_cols], 
                      diabetes['train']['target'].diseaseProgression)
test_acc = dtr.score(diabetes['test']['attributes'][input_cols], 
                     diabetes['test']['target'].diseaseProgression)

plt.title(f"Decision Tree Regressor (max_depth=3)\nTraining: {train_acc:.3f}, Testing: {test_acc:.3f}")
plt.xlabel(input_cols[0])
plt.ylabel(input_cols[1])
plt.legend()

plt.show()
