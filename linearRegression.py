import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.datasets import fetch_california_housing

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
print(np.array([5, 15, 25, 35, 45, 55]))
print(x)
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept: {new_model.intercept_}")
print(f"slope: {new_model.coef_}")
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")

# Multiple Linear Regression With scikit-learn
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
print(x)
print(y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print(f"predicted response:\n{y_pred}")

from sklearn.preprocessing import PolynomialFeatures
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
print(x_)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")
y_pred = model.predict(x_)
print(f"predicted response:\n{y_pred}")

 
california_housing = fetch_california_housing(as_frame=True) 
#We can have a first look at the available description
print(california_housing.DESCR)
#overview of the entire dataset
california_housing.frame.head()
#to display only the features used by a predictive model (without target)
california_housing.data.head()
# to check  the data types and if the dataset contains any missing value
california_housing.frame.info()
# to see this specificity looking at the statistics for these features:
california= pd.DataFrame(california_housing.data,columns=california_housing.feature_names)
california['MedHouseVal'] = california_housing.target
plt.figure(figsize=(20, 5))
features = ['HouseAge', 'AveRooms']
target = california['MedHouseVal’]
for i, col in enumerate(features): 
    plt.subplot(1, len(features) , i+1) 
    x = california[col]
    y = target 
    plt.scatter(x, y, marker='o') 
    plt.title(col) 
    plt.xlabel(col) 
    plt.ylabel('MedHouseVal')
X = pd.DataFrame(np.c_[california['HouseAge'], california['AveRooms']], columns = ['HouseAge','AveRooms']) 
Y = california['MedHouseVal']
#Training Model
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=5) 
print(X_train.shape, X_test.shape, Y_train.shape,Y_test.shape)
lin_model = LinearRegression() 
lin_model.fit(X_train, Y_train)
# model evaluation for training set 
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict))) 
r2 = r2_score(Y_train, y_train_predict)
print("The model performance for training set") 
print("--------------------------------------") 
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2)) 
print("\n")





from sklearn import linear_model
#X represents the size of a tumor in centimeters.
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

#Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
#y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)
#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(np.array([3.46]).reshape(-1,1))
print(predicted)

log_odds = logr.coef_
odds = np.exp(log_odds)
print(odds)


