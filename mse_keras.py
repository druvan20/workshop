from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#print('hi')
california=fetch_california_housing()


X_train,X_test,y_train,y_test=train_test_split(california.data,california.target,test_size=0.2)
#normalized the data
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#train a model
model=LinearRegression()
model.fit(X_train,y_train)


#predict
pred=model.predict(X_test)
mse=mean_squared_error(pred,y_test)
print(mse)
print('hi')

plt.scatter(X_test,y_test,label='original data')


#mlp
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from keras.utils import to_categorical
from keras.datasets import mnist

#load the dataset
(X_train,y_train),(X_test,y_test)=mnist,load_data()
print(X_train.shape,y_train.shape)
