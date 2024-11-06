import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 1)
y = 2 * X+0.005
#print(y)
model = Sequential()
model.add(Dense(2, input_dim=1))  

model.add(Dense(1)) 
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X, y,epochs=500) 

output=model.predict(X)
#print(output)





plt.scatter(X,y,label='original data')
plt.plot(X,output,label='predicted data')
plt.show()






