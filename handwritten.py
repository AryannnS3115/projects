import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os
for dirname, _, filenames in os.walk('D:\hand-written-digit-recognition-using-mnist-dataset.ipynb'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        train_df=pd.read_csv('D:\hand-written-digit-recognition-using-mnist-dataset.ipynb')
test_df=pd.read_csv('D:\hand-written-digit-recognition-using-mnist-dataset.ipynb')
x=train_df.drop('label',axis=1).values
y=train_df['label'].values
plt.bar(list(range(10)),train_df['label'].value_counts())
plt.xticks(list(range(10)))
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.title('Distribution of various digits in Dataset')
plt.show()
fig,axes=plt.subplots(3,6,figsize=(18,11),sharex=True,sharey=True)
for i,axes in enumerate(axes.flat):
    random_digit=np.random.randint(0,y.shape[0])
    axes.set_xlabel(y[random_digit])
    axes.imshow(x[random_digit].reshape(28,28),cmap='gray')from tensorflow import keras
x=x/255
x_reshaped=x.reshape(x.shape[0],28,28,1)
classes_count=len(np.unique(y))
y_categorical=keras.utils.to_categorical(y,classes_count)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_reshaped,y_categorical,test_size=0.2,stratify=y)
x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,test_size=0.2,stratify=y_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adam
model=Sequential()
filters_count=64
filter1_size=(5,5)
filter2_size=(3,3)
pool_kernel_size=(2,2)
node_count=500
model.add(Conv2D(
          filters_count,
          filter1_size,
          input_shape=(x_train.shape[1],x_train.shape[2],1),
          activation='relu'
))
model.add(Conv2D(
          filters_count,
          filter1_size,
          activation='relu'
))
model.add(MaxPooling2D(pool_size=pool_kernel_size))
model.add(Conv2D(
          filters_count//2,
          filter1_size,
          input_shape=(x_train.shape[1],x_train.shape[2],1),
          activation='relu'
))
model.add(Conv2D(
          filters_count//2,
          filter1_size,
          input_shape=(x_train.shape[1],x_train.shape[2],1),
          activation='relu'
))
model.add(MaxPooling2D(pool_size=pool_kernel_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(node_count,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_count,activation='softmax'))
model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics='accuracy')
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
earlyStopping=EarlyStopping(monitor='accuracy',mode='min',verbose=1,patience=50)
checkpoints=ModelCheckpoint("Digit_Recognizer.h5",monitor='val_accuracy',mode='min',save_be
                            history=model.fit(
    x_train,
    y_train,
    batch_size=50,
    epochs=10,
    validation_data=(x_cv,y_cv),
    callbacks=[earlyStopping,checkpoints])plt.plot(history.history['loss'],label='Training_loss')
plt.plot(history.history['accuracy'],label='Training_accuracy')
plt.xlabel('Epochs')
plt.title('Training Loss Vs Accuracy')
plt.legend()
plt.show()
plt.plot(history.history['val_loss'],label='Testing_loss')
plt.plot(history.history['val_accuracy'],label='Testing_accuracy')
plt.xlabel('Epochs')
plt.title('Testing Loss Vs Accuracy')
plt.legend()
plt.show()
from tensorflow.keras.models import load_model
model=load_model('Digit_Recognizer.h5')
_,train_accuracy=model.evaluate(x_train,y_train)
print(f"Training Accuracy: {train_accuracy}")
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.figure(figsize=(12,10))
y_hat=model.predict(x_test)
y_hat=np.argmax(y_hat,axis=1)
y_true=np.argmax(y_test,axis=1)
sns.heatmap(confusion_matrix(y_true,y_hat),annot=True)
plt.show()
fig,axes=plt.subplots(3,6,figsize=(18,11),sharex=True,sharey=True)
for i,axes in enumerate(axes.flat):
    random_digit=np.random.randint(0,x_test.shape[0])
    axes.set_xlabel(f"Predicted: {np.argmax(model.predict(np.expand_dims(x_test[random_digit],0)))}")
    axes.set_ylabel(f"Actual: {np.argmax(y_test[random_digit])}")
    axes.imshow(x_test[random_digit].reshape(28,28))
    test_images=test_df.values
test_images=test_images.reshape(len(test_df),28,28,1)
test_images=test_images/255
y_pred=model.predict(test_images)
y_pred=np.argmax(y_pred,axis=1)
submission=pd.DataFrame({"ImageId":list(range(1,len(y_pred)+1)),"Label": y_pred})
submission.to_csv("submission.csv",index=False,header=True)