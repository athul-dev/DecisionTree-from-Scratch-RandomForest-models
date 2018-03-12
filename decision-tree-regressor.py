import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd

# number_of_samples = 100
# x = np.linspace(-np.pi, np.pi, number_of_samples)
# y = 0.5*x+np.sin(x)+np.random.random(x.shape)
df = pd.read_csv("poly.csv",sep='\t')
# df = pd.read_csv("play.csv",sep='\t')
print(df.head(6))

x=df['X'].values
y=df['Y'].values

# x=df['Temperature'].values
# y=df['Play/Not'].values

plt.scatter(x,y,color='black') #Plot y-vs-x in dots
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title(' Decision Tree')
plt.show()

#random_indices = np.random.permutation(number_of_samples)
#Training set
x_train = x
y_train = y
# #Validation set
# x_val = x[random_indices[70:85]]
# y_val = y[random_indices[70:85]]
# #Test set
# x_test = x[random_indices[85:]]
# y_test = y[random_indices[85:]]

maximum_depth_of_tree = np.arange(6) + 1
train_err_arr = []
val_err_arr = []
test_err_arr = []

for depth in maximum_depth_of_tree:
    model = tree.DecisionTreeRegressor(max_depth=depth)
    # sklearn takes the inputs as matrices. Hence we reshpae the arrays into column matrices
    x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train), 1))
    y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train), 1))

    # Fit the line to the training data
    model.fit(x_train_for_line_fitting, y_train_for_line_fitting)

    # Plot the line
    plt.figure()
    plt.scatter(x_train, y_train, color='black')
    plt.plot(x.reshape((len(x), 1)), model.predict(x.reshape((len(x), 1))), color='blue')
    plt.xlabel('x-input feature')
    plt.ylabel('y-target values')
    plt.title('Line fit to training data with max_depth=' + str(depth))
    plt.show()