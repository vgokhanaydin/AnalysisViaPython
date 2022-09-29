#Tensorflow Obesity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_excel("dataimport.xlsx")
print(dataFrame.describe())
print(dataFrame.isnull().sum())

#ÖN İNCELEME

#Y'nin Dağılım Grafiği
plt.figure(dpi=300)
sbn.distplot(dataFrame["Y"])
plt.show()

#Korelasyonlar
print(dataFrame.corr())
#Sadece Y'nin korelasyonu ve Sıralanmış Hali
print(dataFrame.corr()["Y"].sort_values())

#ScatterPlot
plt.figure(dpi=300)
sbn.scatterplot(x="X1", y="Y", data=dataFrame)
plt.show()

#Y'ni küçükten büyüğe sıralama
print(dataFrame.sort_values("Y", ascending=True).head(20))

#MODEL OLUŞTURMA

#x ve y leri ayırma
y = dataFrame["Y"].values
x = dataFrame.drop("Y", axis=1).values

#veri setini bölme
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=7)


#verileri scale etme
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#tensorflow model kurma
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#modele nöron koyarken önce bağımsız değişken sayısı kadar koymak gerekir duruma göre yükseltilebilir
model = Sequential()
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test), epochs=120)
#model.fit'in içine batch_size=250 yazarsak verileri 250'lik grup grup modele verir. çok büyük verilerde kullanılabilir.

kayipVerisi = pd.DataFrame(model.history.history)

kayipVerisi.plot()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tahminDizisi = model.predict(x_test)

y_test = y_test.reshape(-1,1)

MAE = mean_absolute_error(y_test, tahminDizisi)
MSE = mean_squared_error(y_test, tahminDizisi)
R2 = r2_score(y_test, tahminDizisi)

print(MAE)
print(MSE)
print(R2)

plt.figure(dpi=300)
plt.scatter(y_test, tahminDizisi)
plt.show()























