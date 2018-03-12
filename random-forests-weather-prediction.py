import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

features = pd.read_csv('temps.csv',sep='\t')
#print(features.head(5))

features = pd.get_dummies(features)
# print(features.head(5))

label=np.array(features['actual'])
#print(label)

features= features.drop('actual', axis = 1)
features_list=list(features.columns)
#print(features_list)

#here we are splitting the data using sklearn
train_features, test_features, train_labels, test_labels = train_test_split(features, label, test_size = 0.25, random_state = 42)

#Training our model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)

#Testing our model
predictions = rf.predict(test_features)

#In orde to check our error, we subtract our test labels what we stored to te predictions made
errors = abs(predictions - test_labels)

print('Mean Absolute Error:', round(np.mean(errors)), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

####################################################################################3
#Visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = features_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')

# SMALL TREE
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)

# Extract the small tree
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = features_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png('small_tree.png')