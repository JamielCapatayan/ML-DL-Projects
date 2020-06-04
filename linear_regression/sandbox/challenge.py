# challenge: use linear regression to create a line of best fit and print out the errors
# credits to scikit documentation & Siraj Raval base code

# import libraries
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =====================================
# ===== Data cleaning and reading =====
# =====================================

# read and clean data
with open('challenge_dataset.txt', 'r') as file:
    data = file.read()

data = data.replace('\n', ' ')
data = data.split(' ')

# create dictionary to store x and y x_values
d = { 'x':None,
      'y':None }
# format data to store in a list
x_values = []
y_values = []
for pair in data:
    if(pair == ''):
        continue
    else:
        list_pair = pair.split(',')
        x = float(list_pair[0])
        y = float(list_pair[1])
        x_values.append(x)
        y_values.append(y)
# update dictionary with lists
d.update(x = x_values)
d.update(y = y_values)
# use pandas to format into dataframes
dataframe = pd.DataFrame(data=d)
data_x = dataframe[['x']]
data_y = dataframe[['y']]

# ===================================
# ==== Data training and testing ====
# ===================================

# split into training and testing sets
data_x_train = data_x[:-60]
data_x_test = data_x[-30:]

data_y_train = data_y[:-60]
data_y_test = data_y[-30:]

# create linear regression object
regr = linear_model.LinearRegression()

# train model with training sets
regr.fit(data_x_test, data_y_test)

# create predictions with the testing sets
data_y_pred = regr.predict(data_x_test)

# Coefficients
print('Coefficients: \n', regr.coef_)
# Mean Squared error
print('Mean squared error: %.2f' % mean_squared_error(data_y_test, data_y_pred))
# Coefficient of Determination
print('Coefficient of determination: %.2f' % r2_score(data_y_test, data_y_pred))

# =======================
# ==== Data plotting ====
# =======================

plt.scatter(data_x_test, data_y_test, color='black')
plt.plot(data_x_test, data_y_pred, color='blue')

plt.xticks(())
plt.yticks(())

plt.show()
