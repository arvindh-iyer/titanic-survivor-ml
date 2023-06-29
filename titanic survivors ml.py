import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu,sigmoid
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import BinaryCrossentropy

books=pd.read_csv('train.csv')
#print(books.shape)
#print(books.keys())
#pd.set_option('display.max_columns', None)
#print(books.head(1))


np_array=books.values

#print(np_array.shape)

x_train=np.delete(np_array,[0,3,8,10],axis=1)

x_train[:,-1]=[2 if i=='C' else  1 if i=='S' else 0 for i in x_train[:,-1]]
print(x_train[:,2])
x_train[:,2]=[1 if i=="male" else 0 for i in x_train[:,2]]
print(x_train[:,2])

print(x_train[0])
x_train=x_train.astype(float)

#print(x_train[0])

empty_rows=np.isnan(x_train).any(axis=1)
#print(empty_rows)
x_train=x_train[~empty_rows]

y_train=x_train[:,0].reshape((-1,1))
#print(y_train)
#print(x_train.shape)
x_train=np.delete(x_train,[0],axis=1)

#print(x_train[:,1])


#print(x_train[:,-1])

#print(x_train[0])



#print(x_train.shape)

print(x_train[0])
print(x_train[:,1])
x_mean=np.average(x_train,axis=0).reshape((1,-1))
x_std=np.std(x_train,axis=0).reshape((1,-1))
print(x_mean)
print(x_std)
x_train=(x_train-x_mean)/x_std
print(x_train.shape)

model=Sequential([
    Dense(units=16,activation=relu,kernel_regularizer=l2(.1)),
    Dense(units=1,activation=sigmoid,kernel_regularizer=l2(.1))
])

model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(.001)
)

model.fit(x_train,y_train,epochs=50)

#print(model.get_weights(model.get_layer('l1')))

books=pd.read_csv('test.csv')
print(books.keys())
#print(books.shape)

x_test=books.values
passenger_id=x_test[:,0].reshape((-1,1))
passenger_id=passenger_id.astype(int)
x_test=np.delete(x_test,[0,2,7,9],axis=1)
x_test[:,-1]=[2 if i=='C' else  1 if i=='S' else 0 for i in x_test[:,-1]]
#print(x_train[:,2])
x_test[:,1]=[1 if i=="male" else 0 for i in x_test[:,1]]
#print(x_test[0:5])
x_test=x_test.astype(float)
for i in range(x_test.shape[1]):
    for j in range(x_test.shape[0]):
        if np.isnan(x_test[j,i]):
            x_test[j,i]=x_mean[0,i]


x_test=(x_test-x_mean)/x_std
print(x_test[:5])
y_test=model.predict(x_test)
print(y_test)
y_test=[1 if i>=0.5 else 0 for i in y_test]
y_test=np.array(y_test).reshape((-1,1))
y_test=y_test.astype(int)
print(y_test)

output=np.concatenate([passenger_id,y_test],axis=1)
output=output.astype(int)
print(output.dtype)
print(output.shape)

file="output.csv"
header="PassengerId,Survived"
np.savetxt(file,output,delimiter=",",header=header,comments="",fmt='%d')