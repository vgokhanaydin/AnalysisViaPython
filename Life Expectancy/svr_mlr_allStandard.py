import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

# veriler
veriler = pd.read_excel("veriler.xlsx")

x = veriler.iloc[:,1:11].values
y = veriler.iloc[:,:1].values

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x = sc1.fit_transform(x)
sc2 = StandardScaler()
y = sc2.fit_transform(y)

# verilerin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)

# TEST VERİSİ MODELLEMESİ ////////////////////////////////////////////////////

# mlr model / test verisi
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_pred = lin_reg.predict(x_test)


# svr model / test verisi
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_train, y_train)
svr_pred = svr_reg.predict(x_test)
svr_pred = svr_pred.reshape(-1,1)

# egitim verisi OLS
model_linear_train = sm.OLS(lin_reg.predict(x_train), x_train)
print(model_linear_train.fit().summary())

model_svr_train = sm.OLS(svr_reg.predict(x_train), x_train)
print(model_svr_train.fit().summary())

# karsilastirma / test verisi
print("\nKARŞILAŞTIRMA TEST VERİSİ")
# r2
lin_r2 = r2_score(y_test, lin_pred)
print("LinearRegression R2: ", lin_r2)
svr_r2 = r2_score(y_test, svr_pred)
print("SupportVectorRegression R2: ", svr_r2)

# r2 adj
lin_r2_adj = 1-((1-lin_r2)*(22-1)/(22-10-1))
print("LinearRegression R2adj: ", lin_r2_adj)
svr_r2_adj = 1-((1-svr_r2)*(22-1)/(22-10-1))
print("SupportVectorRegression R2adj: ", svr_r2_adj)

# mse mean squared error
def mse(actual, predicted):
    return np.mean(np.square(actual-predicted))
lin_mse = mse(y_test, lin_pred)
print("LinearRegression MSE: ", lin_mse)
svr_mse = mse(y_test, svr_pred)
print("SupportVectorRegression MSE: ", svr_mse)

# rmse - root mean squared error
def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual-predicted)))
lin_rmse = rmse(y_test, lin_pred)
print("LinearRegression RMSE: ", lin_rmse)
svr_rmse = rmse(y_test, svr_pred)
print("SupportVectorRegression RMSE: ", svr_rmse)

# mae - mean absolute error
def mae(actual, predicted):
    return np.mean(np.abs(actual-predicted))
lin_mae = mae(y_test, lin_pred)
print("LinearRegression MAE: ", lin_mae)
svr_mae = mae(y_test, svr_pred)
print("SupportVectorRegression MAE: ", svr_mae)

# mae2 - median absolute error
def median_abs_error(actual, predicted):
    return np.sum(np.median(np.abs(actual - predicted)))
lin_mae2 = median_abs_error(y_test, lin_pred)
print("LinearRegression MAE2: ", lin_mae2)
svr_mae2 = median_abs_error(y_test, svr_pred)
print("SupportVectorRegression MAE2: ", svr_mae2)

# mape - mean absolute percentage error
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
lin_mape = mape(y_test, lin_pred)
print("LinearRegression MAPE: ", lin_mape)
svr_mape = mape(y_test, svr_pred)
print("SupportVectorRegression MAPE: ", svr_mape)

# rae - relative absolute error
def rae(actual, predicted):
    numerator = np.sum(np.abs(predicted - actual))
    denominator = np.sum(np.abs(np.mean(actual) - actual))
    return numerator / denominator
lin_rae = rae(y_test, lin_pred)
print("LinearRegression RAE: ", lin_rae)
svr_rae = rae(y_test, svr_pred)
print("SupportVectorRegression RAE: ", svr_rae)

# EĞİTİM VERİSİ MODELLEMESİ ////////////////////////////////////////////////////

# mlr model / eğitim verisi
from sklearn.linear_model import LinearRegression
lin_reg_train = LinearRegression()
lin_reg_train.fit(x_train, y_train)
lin_pred_train = lin_reg_train.predict(x_train)


# svr model / eğitim verisi
from sklearn.svm import SVR
svr_reg_train = SVR(kernel = "rbf")
svr_reg_train.fit(x_train, y_train)
svr_pred_train = svr_reg_train.predict(x_train)
svr_pred_train = svr_pred_train.reshape(-1,1)

# karsilastirma / eğitim verisi
print("\nKARŞILAŞTIRMA EĞİTİM VERİSİ")
# r2
lin_r2_train = r2_score(y_train, lin_pred_train)
print("LinearRegression R2: ", lin_r2_train)
svr_r2_train = r2_score(y_train, svr_pred_train)
print("SupportVectorRegression R2: ", svr_r2_train)

# r2 adj
lin_r2_adj_train = 1-((1-lin_r2_train)*(42-1)/(42-10-1))
print("LinearRegression R2adj: ", lin_r2_adj_train)
svr_r2_adj_train = 1-((1-svr_r2_train)*(42-1)/(42-10-1))
print("SupportVectorRegression R2adj: ", svr_r2_adj_train)

# mse mean squared error
lin_mse_train = mse(y_train, lin_pred_train)
print("LinearRegression MSE: ", lin_mse_train)
svr_mse_train = mse(y_train, svr_pred_train)
print("SupportVectorRegression MSE: ", svr_mse_train)

# rmse - root mean squared error
lin_rmse_train = rmse(y_train, lin_pred_train)
print("LinearRegression RMSE: ", lin_rmse_train)
svr_rmse_train = rmse(y_train, svr_pred_train)
print("SupportVectorRegression RMSE: ", svr_rmse_train)

# mae - mean absolute error
lin_mae_train = mae(y_train, lin_pred_train)
print("LinearRegression MAE: ", lin_mae_train)
svr_mae_train = mae(y_train, svr_pred_train)
print("SupportVectorRegression MAE: ", svr_mae_train)

# mae2 - median absolute error
lin_mae2_train = median_abs_error(y_train, lin_pred_train)
print("LinearRegression MAE2: ", lin_mae2_train)
svr_mae2_train = median_abs_error(y_train, svr_pred_train)
print("SupportVectorRegression MAE2: ", svr_mae2_train)

# mape - mean absolute percentage error
lin_mape_train = mape(y_train, lin_pred_train)
print("LinearRegression MAPE: ", lin_mape_train)
svr_mape_train = mape(y_train, svr_pred_train)
print("SupportVectorRegression MAPE: ", svr_mape_train)

# rae - relative absolute error
lin_rae_train = rae(y_train, lin_pred_train)
print("LinearRegression RAE: ", lin_rae_train)
svr_rae_train = rae(y_train, svr_pred_train)
print("SupportVectorRegression RAE: ", svr_rae_train)

# TÜM VERİ MODELLEMESİ ////////////////////////////////////////////////////

# mlr model / tüm veri
from sklearn.linear_model import LinearRegression
lin_reg_all = LinearRegression()
lin_reg_all.fit(x, y)
lin_pred_all = lin_reg_all.predict(x)


# svr model / tüm veri
from sklearn.svm import SVR
svr_reg_all = SVR(kernel = "rbf")
svr_reg_all.fit(x, y)
svr_pred_all = svr_reg_all.predict(x)
svr_pred_all = svr_pred_all.reshape(-1,1)

# karsilastirma / tüm veri
print("\nKARŞILAŞTIRMA TÜM VERİ")
# r2
lin_r2_all = r2_score(y, lin_pred_all)
print("LinearRegression R2: ", lin_r2_all)
svr_r2_all = r2_score(y, svr_pred_all)
print("SupportVectorRegression R2: ", svr_r2_all)

# r2 adj
lin_r2_adj_all = 1-((1-lin_r2_all)*(64-1)/(64-10-1))
print("LinearRegression R2adj: ", lin_r2_adj_all)
svr_r2_adj_all = 1-((1-svr_r2_all)*(64-1)/(64-10-1))
print("SupportVectorRegression R2adj: ", svr_r2_adj_all)

# mse mean squared error
lin_mse_all = mse(y, lin_pred_all)
print("LinearRegression MSE: ", lin_mse_all)
svr_mse_all = mse(y, svr_pred_all)
print("SupportVectorRegression MSE: ", svr_mse_all)

# rmse - root mean squared error
lin_rmse_all = rmse(y, lin_pred_all)
print("LinearRegression RMSE: ", lin_rmse_all)
svr_rmse_all = rmse(y, svr_pred_all)
print("SupportVectorRegression RMSE: ", svr_rmse_all)

# mae - mean absolute error
lin_mae_all = mae(y, lin_pred_all)
print("LinearRegression MAE: ", lin_mae_all)
svr_mae_all = mae(y, svr_pred_all)
print("SupportVectorRegression MAE: ", svr_mae_all)

# mae2 - median absolute error
lin_mae2_all = median_abs_error(y, lin_pred_all)
print("LinearRegression MAE2: ", lin_mae2_all)
svr_mae2_all = median_abs_error(y, svr_pred_all)
print("SupportVectorRegression MAE2: ", svr_mae2_all)

# mape - mean absolute percentage error
lin_mape_all = mape(y, lin_pred_all)
print("LinearRegression MAPE: ", lin_mape_all)
svr_mape_all = mape(y, svr_pred_all)
print("SupportVectorRegression MAPE: ", svr_mape_all)

# rae - relative absolute error
lin_rae_all = rae(y, lin_pred_all)
print("LinearRegression RAE: ", lin_rae_all)
svr_rae_all = rae(y, svr_pred_all)
print("SupportVectorRegression RAE: ", svr_rae_all)

# grafikler test
plt.figure(figsize=(5,5))
plt.scatter(y_test, lin_pred, s=100, marker="^", c="crimson")
p1 = max(max(lin_pred), max(y_test))
p2 = min(min(lin_pred), min(y_test))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("MLR Test Data", fontsize=18)
plt.xlabel("Actual Values", fontsize=18)
plt.ylabel("Predicted Values", fontsize=18)
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(y_test, svr_pred, s=100, marker="^", c="crimson")
p1 = max(max(svr_pred), max(y_test))
p2 = min(min(svr_pred), min(y_test))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("SVR(rbf) Test Data", fontsize=18)
plt.xlabel("Actual Values", fontsize=18)
plt.ylabel("Predicted Values", fontsize=18)
plt.show()

# grafikler train
plt.figure(figsize=(5,5))
plt.scatter(y_train, lin_reg.predict(x_train), s=100, marker="^", c="crimson")
p1 = max(max(lin_reg.predict(x_train)), max(y_train))
p2 = min(min(lin_reg.predict(x_train)), min(y_train))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("MLR Train Data", fontsize=18)
plt.xlabel("Actual Values", fontsize=18)
plt.ylabel("Predicted Values", fontsize=18)
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(y_train, svr_reg.predict(x_train), s=100, marker="^", c="crimson")
p1 = max(max(svr_reg.predict(x_train)), max(y_train))
p2 = min(min(svr_reg.predict(x_train)), min(y_train))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("SVR(rbf) Train Data", fontsize=18)
plt.xlabel("Actual Values", fontsize=18)
plt.ylabel("Predicted Values", fontsize=18)
plt.show()

# grafikler / tüm veri
plt.figure(figsize=(5,5))
plt.scatter(y, lin_reg_all.predict(x), s=100, marker="^", c="crimson")
p1 = max(max(lin_reg_all.predict(x)), max(y))
p2 = min(min(lin_reg_all.predict(x)), min(y))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("MLR All Data", fontsize=18)
plt.xlabel("Actual Values", fontsize=18)
plt.ylabel("Predicted Values", fontsize=18)
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(y, svr_reg_all.predict(x), s=100, marker="^", c="crimson")
p1 = max(max(svr_reg_all.predict(x)), max(y))
p2 = min(min(svr_reg_all.predict(x)), min(y))
plt.plot([p1,p2], [p1,p2], "b-")
plt.title("SVR(rbf) All Data", fontsize=18)
plt.xlabel("Actual Values", fontsize=18)
plt.ylabel("Predicted Values", fontsize=18)
plt.show()

# özel karma grafik
# svr model / test verisi / polynomial
svr_reg_poly = SVR(kernel = "poly")
svr_reg_poly.fit(x_train, y_train)
svr_pred_poly = svr_reg_poly.predict(x_test)
svr_pred_poly = svr_pred_poly.reshape(-1,1)

plot_data = pd.read_excel("plot.xlsx")
plt.figure(figsize=(10,5), dpi=300)
plot_y = plot_data.iloc[:,:1].values
plot_rbf = plot_data.iloc[:,1:2].values
plot_poly = plot_data.iloc[:,2:3].values
plot_mlr = plot_data.iloc[:,3:4].values
plot_dummy = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
plt.plot(plot_dummy, plot_y, linestyle="-", c="red", marker="s", markersize=7, label="Actual Y")
plt.plot(plot_dummy, plot_rbf, linestyle="--", c="blue", marker="o", markersize=7, label="SVR(rbf) Predicted Y")
#plt.plot(plot_dummy, plot_poly, linestyle="--", c="green")
#plt.plot(plot_dummy, plot_mlr, linestyle=":", c="orange")
plt.title("Predicted Values and Actual Values (Test Data/SVR(RFB))", fontsize=12)
plt.legend(fontsize=10)
plt.show()