from DE import*
from PSO import*
from GA import*
from Objective import*
from Bounds import*
from RandomInitializer import*
from QuasiRandomInitializer import*
from SphereInitializer import*
from LinearInertia import*
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
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
obj=Objective(x_train,y_train,x_valid,y_valid,epochs=600,batch_size=250,loss=loss_fn,patience=50)
npart = 20
ndim = 2
m =50
tol = -500
b = Bounds([0,0], [1,1],enforce="resample")
i = QuasiRandomInitializer(npart, ndim, bounds=b)
t = LinearInertia()
swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol,
max_iter=m, bounds=b,ring=True,neighbors=4,minnodes=30,maxnodes=100,runspernet=4)
opt=swarm.Optimize()
res=swarm.Results()
pos = res["gpos"][-1]
g = res["gbest"][-1]
worst_val_mapes=res["worst_val_mapes"]
mean_val_mapes=res["mean_val_mapes"]
best_val_mapes=res["best_val_mapes"]
worst_train_mapes=res["worst_train_mapes"]
mean_train_mapes=res["mean_train_mapes"]
best_train_mapes=res["best_train_mapes"]


def plotworst(val_mapes, train_mapes):
    iterations = np.linspace(0, len(val_mapes) - 1, len(train_mapes))
    plt.plot(iterations, val_mapes, 'g')
    plt.plot(iterations, train_mapes, 'r')
    plt.legend(['Validation MAE', 'Train MAE'])
    plt.xlabel('Generation')
    plt.xticks(np.arange(0, len(iterations), 5))
    plt.title("Worst swarm MAES over generations")
    plt.savefig("figPSO1.png")
    plt.close()


def plotmean(val_mapes, train_mapes):
    iterations = np.linspace(0, len(val_mapes) - 1, len(train_mapes))
    plt.plot(iterations, val_mapes, 'g')
    plt.plot(iterations, train_mapes, 'r')
    plt.legend(['Validation MAE', 'Train MAE'])
    plt.xlabel('Generation')
    plt.xticks(np.arange(0, len(iterations), 5))
    plt.title("Mean swarm MAES over generations")
    plt.savefig("figPSO2.png")
    plt.close()


def plotbest(val_mapes, train_mapes):
    iterations = np.linspace(0, len(val_mapes) - 1, len(train_mapes))
    plt.plot(iterations, val_mapes, 'g')
    plt.plot(iterations, train_mapes, 'r')
    plt.legend(['Validation MAE', 'Train MAE'])
    plt.xlabel('Generation')
    plt.xticks(np.arange(0, len(iterations), 5))
    plt.title("Best swarm MAES over generations")
    plt.savefig("figPSO3.png")
    plt.close()

print("Best MAPE:",g)
print("Best structure:",pos)
plotworst(worst_val_mapes,worst_train_mapes)
plotmean(mean_val_mapes,mean_train_mapes)
plotbest(best_val_mapes,best_train_mapes)















