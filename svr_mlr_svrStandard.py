import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import statsmodels.api as sm

# veriler
veriler = pd.read_excel("veriler.xlsx")

x_linear = veriler.iloc[:,1:11].values
y_linear = veriler.iloc[:,:1].values

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x_linear)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y_linear)

# verilerin bolunmesi
from sklearn.model_selection import train_test_split
x_train_linear, x_test_linear, y_train_linear, y_test_linear = train_test_split(x_linear, y_linear, test_size=0.33, random_state=7)
x_train_svr, x_test_svr, y_train_svr, y_test_svr = train_test_split(x_olcekli, y_olcekli, test_size=0.33, random_state=7)

# mlr model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train_linear, y_train_linear)
lin_pred = lin_reg.predict(x_test_linear)


# svr model
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_train_svr, y_train_svr)
svr_pred = svr_reg.predict(x_test_svr)
svr_pred = svr_pred.reshape(-1,1)


# grafikler test
plt.figure(figsize=(7,7))
plt.scatter(y_test_linear, lin_pred, c="crimson")
p1 = max(max(lin_pred), max(y_test_linear))
p2 = min(min(lin_pred), min(y_test_linear))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("MLR Test Verileri")
plt.xlabel("Gerçek Veriler")
plt.ylabel("Tahmini Veriler")
plt.show()

plt.figure(figsize=(7,7))
plt.scatter(y_test_svr, svr_pred, c="crimson")
p1 = max(max(svr_pred), max(y_test_svr))
p2 = min(min(svr_pred), min(y_test_svr))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("SVR Test Verileri")
plt.xlabel("Gerçek Veriler")
plt.ylabel("Tahmini Veriler")
plt.show()

# egitim verisi OLS
model_linear_train = sm.OLS(lin_reg.predict(x_train_linear), x_train_linear)
print(model_linear_train.fit().summary())

model_svr_train = sm.OLS(svr_reg.predict(x_train_svr), x_train_svr)
print(model_svr_train.fit().summary())

# grafikler train
plt.figure(figsize=(7,7))
plt.scatter(y_train_linear, lin_reg.predict(x_train_linear), c="crimson")
p1 = max(max(lin_reg.predict(x_train_linear)), max(y_train_linear))
p2 = min(min(lin_reg.predict(x_train_linear)), min(y_train_linear))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("MLR Eğitim Verileri")
plt.xlabel("Gerçek Veriler")
plt.ylabel("Tahmini Veriler")
plt.show()

plt.figure(figsize=(7,7))
plt.scatter(y_train_svr, svr_reg.predict(x_train_svr), c="crimson")
p1 = max(max(svr_reg.predict(x_train_svr)), max(y_train_svr))
p2 = min(min(svr_reg.predict(x_train_svr)), min(y_train_svr))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("SVR Eğitim Verileri")
plt.xlabel("Gerçek Veriler")
plt.ylabel("Tahmini Veriler")
plt.show()

# karsilastirma
# r2
lin_r2 = r2_score(y_test_linear, lin_pred)
print("LinearRegression R2: ", lin_r2)
svr_r2 = r2_score(y_test_svr, svr_pred)
print("SupportVectorRegression R2: ", svr_r2)

# r2 adj
lin_r2_adj = 1-((1-lin_r2)*(22-1)/(22-10-1))
print("LinearRegression R2adj: ", lin_r2_adj)
svr_r2_adj = 1-((1-svr_r2)*(22-1)/(22-10-1))
print("SupportVectorRegression R2adj: ", svr_r2_adj)

# mse mean squared error
def mse(actual, predicted):
    return np.mean(np.square(actual-predicted))
lin_mse = mse(y_test_linear, lin_pred)
print("LinearRegression MSE: ", lin_mse)
svr_mse = mse(y_test_svr, svr_pred)
print("SupportVectorRegression MSE: ", svr_mse)

# rmse - root mean squared error
def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual-predicted)))
lin_rmse = rmse(y_test_linear, lin_pred)
print("LinearRegression RMSE: ", lin_rmse)
svr_rmse = rmse(y_test_svr, svr_pred)
print("SupportVectorRegression RMSE: ", svr_rmse)

# mae - mean absolute error
def mae(actual, predicted):
    return np.mean(np.abs(actual-predicted))
lin_mae = mae(y_test_linear, lin_pred)
print("LinearRegression MAE: ", lin_mae)
svr_mae = mae(y_test_svr, svr_pred)
print("SupportVectorRegression MAE: ", svr_mae)

# mae2 - median absolute error
def median_abs_error(actual, predicted):
    return np.sum(np.median(np.abs(actual - predicted)))
lin_mae2 = median_abs_error(y_test_linear, lin_pred)
print("LinearRegression MAE2: ", lin_mae2)
svr_mae2 = median_abs_error(y_test_svr, svr_pred)
print("SupportVectorRegression MAE2: ", svr_mae2)

# mape - mean absolute percentage error
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
lin_mape = mape(y_test_linear, lin_pred)
print("LinearRegression MAPE: ", lin_mape)
svr_mape = mape(y_test_svr, svr_pred)
print("SupportVectorRegression MAPE: ", svr_mape)

# rae - relative absolute error
def rae(actual, predicted):
    numerator = np.sum(np.abs(predicted - actual))
    denominator = np.sum(np.abs(np.mean(actual) - actual))
    return numerator / denominator
lin_rae = rae(y_test_linear, lin_pred)
print("LinearRegression RAE: ", lin_rae)
svr_rae = rae(y_test_svr, svr_pred)
print("SupportVectorRegression RAE: ", svr_rae)




























