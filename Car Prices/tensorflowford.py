import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_excel("forddata.xlsx")
print(df.describe())
print(df.isnull().sum())

#Fiyat Dağılım Grafiği
#Dağılım 1
sbn.set(font_scale=1.4)

plt.figure(figsize=(10,7))
plt.suptitle("Fiyat Dağılımı/Price Distribution")
sbn.distplot(df["Fiyat"])
plt.xlabel("Fiyat/Price")
plt.ylabel("Yoğunluk/Intensity")
plt.show()

#Dağılım 2
plt.figure(figsize=(15,7))
plt.suptitle("Fiyat Dağılımı/Price Distribution")

plt.subplot(1,2,1)
sbn.distplot(x=df.Fiyat)
plt.ylabel("Yoğunluk/Intensity")

plt.subplot(1,2,2)
sbn.boxenplot(y=df.Fiyat)
plt.ylabel("Fiyat/Price")
plt.show()

#Aykırı Değerlerden Kurtulma
len = ['Fiyat','Mil','Vergi','Yakıt/Mil']
for i in len:
   x = df[i].describe()
   Q1 = x[4]
   Q3 = x[6]
   IQR = Q3-Q1
   lower_bound = Q1-(1.5*IQR)
   upper_bound = Q3+(1.5*IQR)
   df = df[(df[i]>lower_bound)&(df[i]<upper_bound)]
   
#Kategorik Değişkenlerin Kodlanması
vitesdummy = pd.get_dummies(df.Vites)
yakıtdummy = pd.get_dummies(df.Yakıt)

#Kodlanmış Kategorik Değişkenlerin Veri Setine Eklenmesi
df2 = pd.concat([df,vitesdummy,yakıtdummy], axis=1)

#Tekrar Eden Kategoriklerin Çıkarılması
df2 = df2.drop(["Vites","Yakıt"], axis=1)

df = df2
df = df.reset_index(drop=True)

#Verilerin Scale Edilmesi
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
inputs = ["Yıl","Fiyat","Mil","Vergi","Yakıt/Mil","Motor"]
df[inputs] = mms.fit_transform(df[inputs])


#Korelasyon Grafiği
plt.figure(figsize = (50, 30))
sbn.set(font_scale=3)
x_axis_labels = ["Yıl/Year", "Fiyat/Price","Mil/Mile","Vergi/Tax","Yakıt/Mil-Fuel/Mile","Motor/Engine"]
y_axis_labels = ["Yıl/Year", "Fiyat/Price","Mil/Mile","Vergi/Tax","Yakıt/Mil-Fuel/Mile","Motor/Engine"]
sbn.heatmap(df[["Yıl","Fiyat","Mil","Vergi","Yakıt/Mil","Motor"]].corr(), annot = True, vmin=-1, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.show()

#Aykırı gözlemler çıkarılmış dağılım grafiği
plt.figure(figsize=(15,7))
plt.suptitle("Fiyat Dağılımı/Price Distribution")

plt.subplot(1,2,1)
sbn.distplot(x=df.Fiyat)
plt.ylabel("Yoğunluk/Intensity")

plt.subplot(1,2,2)
sbn.boxenplot(y=df.Fiyat)
plt.ylabel("Fiyat/Price")
plt.show()

#X ve Y lerin ayrılması
x = df.drop("Fiyat", axis=1)
y = df.Fiyat

#Train ve Test data ayrılması
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=7)

#Tensorflow Model Kurulumu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=x_train, y=y_train, batch_size=250, validation_data=(x_test,y_test), epochs=3000)

kayipVerisi = pd.DataFrame(model.history.history)

sbn.set(font_scale=1.5)
plt.rcParams["figure.figsize"]=[10,7]
#plt.rcParams["figure.autolayout"]=True
plt.rcParams["figure.dpi"]=300
kayipVerisi.plot(figsize=(10,7))
plt.show()



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tahminDizisi = model.predict(x_test)

#Linear Model Kurulumu
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_pred = lin_reg.predict(x_test)

#SVR Model Kurulumu
from sklearn.svm import SVR
svr_reg = SVR(kernel = "poly", degree=7)
svr_reg.fit(x_train, y_train)
svr_pred = svr_reg.predict(x_test)

#Decision Tree Model Kurulumu
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(x_train, y_train)
dt_pred = dt_reg.predict(x_test)

#Karşılaştırmaların Hesaplanması
MAE = mean_absolute_error(y_test, tahminDizisi)
MSE = mean_squared_error(y_test, tahminDizisi)
R2 = r2_score(y_test, tahminDizisi)

MAE_lin = mean_absolute_error(y_test, lin_pred)
MSE_lin = mean_squared_error(y_test, lin_pred)
R2_lin = r2_score(y_test, lin_pred)

MAE_svr = mean_absolute_error(y_test, svr_pred)
MSE_svr = mean_squared_error(y_test, svr_pred)
R2_svr = r2_score(y_test, svr_pred)

MAE_dt = mean_absolute_error(y_test, dt_pred)
MSE_dt = mean_squared_error(y_test, dt_pred)
R2_dt = r2_score(y_test, dt_pred)


print("Tensorflow MAE: ", MAE)
print("Tensorflow MSE: ", MSE)
print("Tensorflow R2: ", R2)

print("Linear MAE: ", MAE_lin)
print("Linear MSE: ", MSE_lin)
print("Linear R2: ", R2_lin)

print("SVR MAE: ", MAE_svr)
print("SVR MSE: ", MSE_svr)
print("SVR R2: ", R2_svr)

print("DT MAE: ", MAE_dt)
print("DT MSE: ", MSE_dt)
print("DT R2: ", R2_dt)

#Gerçek ve Tahmin Grafikleri
plt.figure(figsize=(10,7), dpi=300)
plt.scatter(y_test, tahminDizisi)
plt.suptitle("TensorFlow")
p1 = max(max(tahminDizisi), max(y_test))
p2 = min(min(tahminDizisi), min(y_test))
plt.plot([p1,p2], [p1,p2], "green")
plt.xlabel("Gerçek/Measurement")
plt.ylabel("Tahmin/Prediction")
plt.show()

plt.figure(figsize=(10,7), dpi=300)
plt.scatter(y_test, lin_pred)
plt.suptitle("Çoklu Doğrusal Regresyon/Multiple Linear Regression")
p3 = max(max(lin_pred), max(y_test))
p4 = min(min(lin_pred), min(y_test))
plt.plot([p3,p4], [p3,p4], "green")
plt.xlabel("Gerçek/Measurement")
plt.ylabel("Tahmin/Prediction")
plt.show()

plt.figure(figsize=(10,7), dpi=300)
plt.scatter(y_test, svr_pred)
plt.suptitle("Destek Vektör Regresyonu/Support Vector Regression")
p5 = max(max(svr_pred), max(y_test))
p6 = min(min(svr_pred), min(y_test))
plt.plot([p5,p6], [p5,p6], "green")
plt.xlabel("Gerçek/Measurement")
plt.ylabel("Tahmin/Prediction")
plt.show()

plt.figure(figsize=(10,7), dpi=300)
plt.scatter(y_test, dt_pred)
plt.suptitle("Karar Ağacı Regresyon/Decision Tree Regression")
p7 = max(max(dt_pred), max(y_test))
p8 = min(min(dt_pred), min(y_test))
plt.plot([p7,p8], [p7,p8], "green")
plt.xlabel("Gerçek/Measurement")
plt.ylabel("Tahmin/Prediction")
plt.show()