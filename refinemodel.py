import sys
import numpy as np
import pandas as pd
import keras.models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers,models,callbacks,optimizers,metrics
scaler=MinMaxScaler()
from tensorflow import keras
import matplotlib.pyplot as plt
loss_fn=keras.losses.MeanAbsoluteError()
newdata=[]
data=pd.read_csv("KAG_energydata_complete.csv")
data=np.array(data)
for i in range(len(data)):
    if data[i][1]>97.694958-3*102.524891 and data[i][1]<97.694958+3*102.524891:
       newdata.append(data[i])
data=newdata
data=np.array(data)
data=data.T
dates=data[0]
data=data[1:]
months=[]
days=[]
hours=[]

def datetonum(date):
    month=date[5:7]
    day=date[8:10]
    hour=(date[11:13])
    hour_decimal=int(date[14:16])/60
    if month[0]==0:
       month=month[1]
    if day[0]==0:
       day=day[1]
    if hour[0]==0:
       hour=hour[1]
    month=int(month)
    day=int(day)
    hour=int(hour)+hour_decimal
    return month,day,hour
for date in dates:
    month,day,hour=datetonum(date)
    months.append(month)
    days.append(day)
    hours.append(hour)
data=np.vstack([data,months])
data=np.vstack([data,days])
data=np.vstack([data,hours])
data=np.asarray(data).astype(np.float32)

data=data.T
y=data[:,0]

x=data[:,1:]
x_train, x_rem, y_train, y_rem = train_test_split(x,y, train_size=0.6,random_state=42)
x_test, x_valid, y_test, y_valid = train_test_split(x_rem,y_rem, train_size=0.625,random_state=42)
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_valid= scaler.transform(x_valid)
x_test = scaler.transform(x_test)

ann=models.load_model('BESTMODEL.h5')
ypred=ann.predict(x_train)
print("BEFORE REFINEMENT")
print("TRAIN")
train_rmse1=np.sqrt(mean_squared_error(y_train,ypred))
train_r21=r2_score(y_train,ypred)
train_mae1=mean_absolute_error(y_train,ypred)
train_mape1=mean_absolute_percentage_error(y_train,ypred)*100
print("RMSE",train_rmse1)
print("R2",train_r21)
print("MAE",train_mae1)
print("MAPE",train_mape1)

print("Validation")
ypred=ann.predict(x_valid)
val_rmse1=np.sqrt(mean_squared_error(y_valid,ypred))
val_r21=r2_score(y_valid,ypred)
val_mae1=mean_absolute_error(y_valid,ypred)
val_mape1=mean_absolute_percentage_error(y_valid,ypred)*100
print("RMSE",val_rmse1)
print("R2",val_r21)
print("MAE",val_mae1)
print("MAPE",val_mape1)

ypred=ann.predict(x_test)
print("TEST")
test_rmse1=np.sqrt(mean_squared_error(y_test,ypred))
test_r21=r2_score(y_test,ypred)
test_mae1=mean_absolute_error(y_test,ypred)
test_mape1=mean_absolute_percentage_error(y_test,ypred)*100
print("RMSE",test_rmse1)
print("R2",test_r21)
print("MAE",test_mae1)
print("MAPE",test_mape1)
ann.compile(optimizer=optimizers.SGD(0.001,momentum=0.9,nesterov=True), loss=loss_fn)
checkpoint_cb = keras.callbacks.ModelCheckpoint("BESTMODEL_REFINED.h5",
 save_best_only=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.75, patience=30)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
history = ann.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                    batch_size=8, epochs=10000, callbacks=[lr_scheduler,early_stopping_cb,checkpoint_cb], verbose=2)
val_losses = np.array(history.history["val_loss"])
train_losses = np.array(history.history["loss"])
index_min = np.argmin(val_losses)
val=val_losses[index_min]
train=train_losses[index_min]
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

ann=models.load_model('BESTMODEL_REFINED.h5')
ypred=ann.predict(x_train)
print("AFTER REFINEMENT")
print("TRAIN")
train_rmse2=np.sqrt(mean_squared_error(y_train,ypred))
train_r22=r2_score(y_train,ypred)
train_mae2=mean_absolute_error(y_train,ypred)
train_mape2=mean_absolute_percentage_error(y_train,ypred)*100
print("RMSE",train_rmse2)
print("R2",train_r22)
print("MAE",train_mae2)
print("MAPE",train_mape2)

print("Validation")
ypred=ann.predict(x_valid)
val_rmse2=np.sqrt(mean_squared_error(y_valid,ypred))
val_r22=r2_score(y_valid,ypred)
val_mae2=mean_absolute_error(y_valid,ypred)
val_mape2=mean_absolute_percentage_error(y_valid,ypred)*100
print("RMSE",val_rmse2)
print("R2",val_r22)
print("MAE",val_mae2)
print("MAPE",val_mape2)

ypred=ann.predict(x_test)
print("TEST")
test_rmse2=np.sqrt(mean_squared_error(y_test,ypred))
test_r22=r2_score(y_test,ypred)
test_mae2=mean_absolute_error(y_test,ypred)
test_mape2=mean_absolute_percentage_error(y_test,ypred)*100
print("RMSE",test_rmse2)
print("R2",test_r22)
print("MAE",test_mae2)
print("MAPE",test_mape2)

def percentage_change(v1,v2):
    return (v2-v1)/v1
print("PERCENTAGE")
print("TRAIN")
print("RMSE",percentage_change(train_rmse1,train_rmse2))
print("R2",percentage_change(train_r21,train_r22))
print("MAE",percentage_change(train_mae1,train_mae2))
print("MAPE",percentage_change(train_mape1,train_mape2))

print("Validation")
print("RMSE",percentage_change(val_rmse1,val_rmse2))
print("R2",percentage_change(val_r21,val_r22))
print("MAE",percentage_change(val_mae1,val_mae2))
print("MAPE",percentage_change(val_mape1,val_mape2))

print("TEST")
print("RMSE",percentage_change(test_rmse1,test_rmse2))
print("R2",percentage_change(test_r21,test_r22))
print("MAE",percentage_change(test_mae1,test_mae2))
print("MAPE",percentage_change(test_mape1,test_mape2))