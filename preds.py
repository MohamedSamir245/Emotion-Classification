import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

Emotions = np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))

#unzip the data file first then edit the directory
df=pd.read_csv("D:\Machine learning\projects\Emotion Classification v2/fer2013.csv")
public_test_df=df[df.Usage=='PublicTest']

x_pred = np.array(list(map(str.split, public_test_df.pixels)), np.float32) 
x_pred = x_pred.reshape(x_pred.shape[0], 48, 48, 1) 
x_pred_scaled=x_pred/255
y_pred=np.array(public_test_df.emotion)

model=tf.keras.models.load_model("D:\Machine learning\projects\Emotion Classification v2\emotion_model")

preds=model.predict(x_pred_scaled)
preds_classes=np.argmax(preds,axis=1)


f, axarr = plt.subplots(3,5, figsize=(20,10))

for i in range(3):
    for j in range(5):
        const = np.random.randint(0, len(y_pred)-15)
        idx = const + i*5+j
        axarr[i,j].imshow(x_pred[idx].squeeze(), cmap='gray')
        axarr[i,j].set_title(f'{Emotions[preds_classes[idx]]} ({Emotions[y_pred[idx]]})', color=('green' if preds_classes[idx]==y_pred[idx] else 'red'))
        axarr[i,j].set_xticks([])
        axarr[i,j].set_yticks([])

plt.show()
