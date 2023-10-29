import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Initialize Data

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")
y = df["logS"]
X = df.drop("logS", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100);

# Linear Regression (LR) Model

lr = LinearRegression() 
lr.fit(X_train, y_train) 

y_lr_train_prediction = lr.predict(X_train)
y_lr_test_prediction = lr.predict(X_test)

# Error Analysis

lr_train_mse = mean_squared_error(y_train, y_lr_train_prediction)
lr_train_r2 = r2_score(y_train, y_lr_train_prediction)

lr_test_mse = mean_squared_error(y_test, y_lr_test_prediction)
lr_test_r2 = r2_score(y_test, y_lr_test_prediction)

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_mse]).transpose()
lr_results.columns = ["Method", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]

# Random Forests (RF) Method

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

y_rf_train_prediction = rf.predict(X_train)
y_rf_test_prediction = rf.predict(X_test)

# Error Analysis

rf_train_mse = mean_squared_error(y_train, y_rf_train_prediction)
rf_train_r2 = r2_score(y_train, y_rf_train_prediction)

rf_test_mse = mean_squared_error(y_test, y_rf_test_prediction)
rf_test_r2 = r2_score(y_test, y_rf_test_prediction)

rf_results = pd.DataFrame(["Random Forests", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_mse]).transpose()
rf_results.columns = ["Method", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]

# Printing Results

df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

print(df_models)

# Data Visualization

z = np.polyfit(y_train, y_lr_train_prediction, 1)
p = np.poly1d(z)

plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_lr_train_prediction, alpha=0.3)

plt.plot(y_train, p(y_train), "#F8766D")
plt.ylabel("Predicted LogS")
plt.xlabel("Experimental LogS")

plt.show()
