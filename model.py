import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

Emotions = np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))
#unzip the data file first then edit the directory
df=pd.read_csv("D:\Machine learning\projects\Emotion Classification v2/fer2013.csv")

print(df.columns)
print(df.emotion.isnull().sum())
print(df.pixels.isnull().sum())
print(df.Usage.isnull().sum())

train_df=df[df.Usage=='Training']
public_test_df=df[df.Usage=='PublicTest']
private_test_df=df[df.Usage=='PrivateTest']

print(train_df.shape)
print(public_test_df.shape)
print(private_test_df.shape)

X_train = np.array(list(map(str.split, train_df.pixels)), np.float32) 
X_test = np.array(list(map(str.split, private_test_df.pixels)), np.float32)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) 
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

y_train=np.array(train_df.emotion)
y_test=np.array(private_test_df.emotion)

X_train_scaled=X_train/255
X_test_scaled=X_test/255

x_pred = np.array(list(map(str.split, public_test_df.pixels)), np.float32) 
x_pred = x_pred.reshape(x_pred.shape[0], 48, 48, 1) 
x_pred_scaled=x_pred/255
y_pred=np.array(public_test_df.emotion)

# data_augmentation = keras.Sequential(
#   [
#     keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
#                                                  input_shape=(48, 
#                                                               48,
#                                                               1)),
#     keras.layers.experimental.preprocessing.RandomRotation(0.1),
#     keras.layers.experimental.preprocessing.RandomZoom(0.1),
#   ]
# )


model=tf.keras.Sequential([
    # data_augmentation,
    
    layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(48,48,1)),
    layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(96,(3,3),dilation_rate=(2, 2),activation='relu',padding='same'),
    layers.Conv2D(96,(3,3),activation='relu',padding='valid'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3),dilation_rate=(2, 2),activation='relu',padding='same'),
    layers.Conv2D(128,(3,3),activation='relu',padding='valid'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(7,activation='sigmoid')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

hist=model.fit(X_train_scaled,y_train,epochs=50,validation_data=(X_train_scaled,y_train),callbacks=lr_reduce)

print(model.evaluate(X_test_scaled,y_test))

preds=model.predict(x_pred_scaled)

# f, axarr = plt.subplots(3,5, figsize=(20,10))
# for i in range(3):
#     for j in range(5):
#         const = np.random.randint(0, len(y_pred)-15)
#         idx = const + i*5+j
#         axarr[i,j].imshow(x_pred[idx].squeeze(), cmap='gray')
#         axarr[i,j].set_title(f'{Emotions[preds[idx]]} ({Emotions[y_pred[idx]]})', color=('green' if preds[idx]==y_pred[idx] else 'red'))
#         axarr[i,j].set_xticks([])
#         axarr[i,j].set_yticks([])
        
# plt.show()

model.save("D:\Machine learning\projects\Emotion Classification v2\emotion_model")