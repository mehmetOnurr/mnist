# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:44:35 2021

@author: Asus
"""

from keras.models import Sequential

from keras.layers import Conv2D,Flatten,MaxPooling2D,Activation,Dropout,Dense,BatchNormalization

from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from keras.utils.np_utils import to_categorical
from glob import glob

import warnings

warnings.filterwarnings('ignore')
#%%   veri yükleme  ve boyut öğrenimi
train_values = pd.read_csv("mnist/mnist_train.csv")

test_values = pd.read_csv("mnist/mnist_test.csv")

print(train_values.shape)
print(train_values.head())


print(test_values.shape)
print(test_values.head())
#%% degerleri yükleme

Y_train = train_values["label"]

X_train = train_values.drop(labels=["label"], axis =1)


Y_test = test_values["label"]

X_test = test_values.drop(labels=["label"],axis =1)


#%% train verilerini grafiğe dökme  ,pallette = sns.color_palette("flare",as_cmap= True)

plt.figure(figsize= (15,7))
"""
g = sns.countplot(data = Y_train)
#plt.hist(x=Y_train,color ="rbg")
plt.title("Number of digit classes")
print(Y_train.value_counts())
"""
# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlBu_r')

# Plot histogram.
n, bins, patches = plt.hist(Y_train, 25, color='green')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title("Number of digit classes")
print(Y_train.value_counts())
plt.show()

#%%   örnek resim

img = X_train.iloc[0].values

img = img.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.title(train_values.iloc[0,0])
plt.axis("off")
plt.show()

#%%

X_train = X_train/255.0

X_test = X_test/255.0

print("X train shape: ",X_train.shape)
print("X Test shape: ",X_test.shape)

print("Y train shape: ",Y_train.shape)
print("Y Test shape: ",Y_test.shape)
#%%

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

print("X train shape: ",X_train.shape)
print("Test shape: ",X_test.shape)

#%% one hot encoder

from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes= 10)

Y_test = to_categorical(Y_test, num_classes= 10)


#%%
plt.imshow(X_train[2][:,:,0], cmap = 'gray')
plt.show()

#%%

numberOfClass = Y_train.shape[1]

model = Sequential()

model.add(Conv2D(filters = 16,kernel_size=(3,3), input_shape= (28,28,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size= (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())
          
model.add(Conv2D(filters=128, kernel_size= (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(units = 256))
model.add(Dropout(0.2))
model.add(Dense(numberOfClass))
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])

hist = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs =25, batch_size = 4000)

#%%
model.save_weights("D:/PycharmProjects/spyderprojeleri/DataiTeam/mnists/deneme2.h5")
#%%

import json
with open('D:/PycharmProjects/spyderprojeleri/DataiTeam/mnists/cnn_mnist_hist.json','w') as f:
    json.dump(hist.history, f)


#%%

import codecs
with codecs.open('D:/PycharmProjects/spyderprojeleri/DataiTeam/mnists/cnn_mnist_hist.json', 'r', encoding= 'utf-8') as f:
    h = json.loads(f.read())



plt.plot(h["loss"], label = "Train loss")
plt.plot(h["val_loss"], label = "Validation loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()

 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    