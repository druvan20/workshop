import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

#load the dataset
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(X_train.shape,X_test.shape)
# plt.imshow(X_train[0])
# plt.show()
print(y_train.shape)

#flatten dimage
# plt.imshow(X_train[0])
X_train=X_train.reshape(X_train.shape[0],784)
X_test=X_test.reshape(X_test.shape[0],784)
print(X_train.shape,X_test.shape)
print(X_train[0])

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
print(y_train.shape)

#normalization
print(X_train.max())
X_train=X_train/255
print(X_train.max())

print(X_test)
X_test=X_test/255
print(X_test)

model=Sequential()
model.add(Dense(10,input_dim=784,activation='softmax'))
#model.add(Dense(8))
model.compile(optimizer=SGD(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
callbacks=ModelCheckpoint('my_mnist.keras',monitor='val_loss',save_best_only=True,mode='min') 
res=model.fit(X_train,y_train,
          epochs=1,
          batch_size=32)
model.evaluate(X_test,y_test)

plt.plot(res.history['loss'])
plt.show()        


