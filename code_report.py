#import package
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statistics as st

#import data
price_demand = pd.read_csv("price_demand_data.csv")
weather_adelaide = pd.read_csv("weather_adelaide.csv")
weather_brisbane = pd.read_csv("weather_brisbane.csv")
weather_melbourne = pd.read_csv("weather_melbourne.csv")
weather_sydney = pd.read_csv("weather_sydney.csv")

#price demand transformation to average by date
price_demand['date']=price_demand['SETTLEMENTDATE'].str.split(" ",expand=True)[0].str.replace(r'\/', '-', regex=True)
price_demand_selected = price_demand.loc[:,['REGION','TOTALDEMAND','date']]
price_demand_bydate = price_demand_selected.groupby(['date','REGION']).agg({'TOTALDEMAND':'mean'}).reset_index().set_index("date")

#Build Histogram to check normality of price demand data
all_demand=price_demand.plot.hist("TOTALDEMAND", title="Total Demand 4 Cities")
pl.xlabel("Amount Energy Demand")
pl.ylabel("Ferquency")
all_demand.figure.savefig("all_demand.png")

#Build Histogram to check normality of price demand data for each city
fig, axs = plt.subplots(2, 2)  # if use subplot
axs = axs.ravel()
unique_city = price_demand["REGION"].unique()
for idx,ax in enumerate(axs):
    ax.hist(price_demand[price_demand["REGION"] == unique_city[idx]]["TOTALDEMAND"])
    ax.set_title(unique_city[idx])
    ax.set_xlabel("Amount Energy Demand")
    ax.set_ylabel("Ferquency")
plt.tight_layout()
fig.savefig("each_country_demand.png")

#adelaide data
#select temperature, humidity, wind speed and pressure features
weather_adelaide_selected=weather_adelaide.loc[:, weather_adelaide.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','3pm wind speed (km/h)','9am wind speed (km/h)'])]
weather_adelaide_selected=weather_adelaide_selected.set_index('Date')

#transform wind speed value
weather_adelaide_selected.loc[weather_adelaide_selected['3pm wind speed (km/h)'] == "Calm",'3pm wind speed (km/h)'] = 1
weather_adelaide_selected.loc[weather_adelaide_selected['9am wind speed (km/h)'] == "Calm",'9am wind speed (km/h)'] = 1
weather_adelaide_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_adelaide_selected['3pm wind speed (km/h)'], errors='ignore')
weather_adelaide_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_adelaide_selected['9am wind speed (km/h)'], errors='ignore')

#exclude outlier
standard_deviations = 3
weather_adelaide_nonoutlier=weather_adelaide_selected[weather_adelaide_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_adelaide_nona=weather_adelaide_nonoutlier.dropna()

#merging total demand to weather condition
adelaide_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "SA1"].merge(weather_adelaide_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
adelaide_corelation = adelaide_merge_data.corr()
adelaide_corelation.to_excel("adelaide_corelation_features.xlsx")

#exclude collinerity in independent feature
adelaide_corelation_selected=adelaide_corelation.loc[adelaide_corelation.index.isin(adelaide_corelation.iloc[0][(adelaide_corelation.iloc[0].abs() > 0.1) ].index),adelaide_corelation.iloc[0][(adelaide_corelation.iloc[0].abs() > 0.1) ].index]
adelaide_corelation_selected=adelaide_corelation_selected.reindex(adelaide_corelation_selected.TOTALDEMAND.abs().sort_values(ascending = False).index)
selected_feauture  = []
for i in range(1,len(adelaide_corelation_selected)) :
  checked_feature = adelaide_corelation_selected.index[i]
  corr_checked = adelaide_corelation_selected.iloc[i]
  if i == 1 :
    selected_feauture.append(checked_feature)
  else :
    if (corr_checked[corr_checked.index.isin(adelaide_corelation_selected.index[1:i])].abs() > 0.5).any() :
      continue
    else :
      selected_feauture.append(checked_feature)

#regression
X=adelaide_merge_data[selected_feauture].reset_index().drop(['index'], axis=1)
y=adelaide_merge_data["TOTALDEMAND"].reset_index().drop(['index'], axis=1)
reg = LinearRegression().fit(X, y)
r_square_adelaide=reg.score(X, y)
coef_regression_adelaide = reg.coef_
intercept_regression_adelaide = reg.intercept_
Y_pred_adelaide = reg.predict(X)
MSE_adelaide = mean_squared_error(y,Y_pred_adelaide)
adelaide_coef = pd.DataFrame([["intercept"]+selected_feauture,intercept_regression_adelaide.tolist() + reg.coef_[0].tolist()],["cofficient name","cofficient value"]).transpose()
adelaide_coef.to_excel("adelaide_model_coefficient.xlsx")

#creating experiment design
# set k=10 value for k-fold CV
k = 10

kf_CV = KFold(n_splits=k, shuffle=True, random_state=42)
results_MSE_adelaide = []
results_r_square_train_adelaide = []
results_r_square_test_adelaide = []

for train_idx, test_idx in kf_CV.split(X):
    # train-test split
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        
    # Training
    reg = LinearRegression().fit(X_train, y_train)
    r_square_train=reg.score(X_train, y_train) 
    r_square_test=reg.score(X_test, y_test)   
    
    # Predictions
    Y_pred = reg.predict(X_test)
    MSE = mean_squared_error(y_test,Y_pred)
    results_MSE_adelaide.append(MSE)
    results_r_square_train_adelaide.append(r_square_train)
    results_r_square_test_adelaide.append(r_square_test)

#making performence metric
performence_metric_adelaide = pd.DataFrame([MSE_adelaide,r_square_adelaide,st.mean(results_MSE_adelaide),st.mean(results_r_square_train_adelaide),st.mean(results_r_square_test_adelaide)],["MSE Original Model","R Square Original Model","MSE 10-CV","R Square Train Model 10-CV","R Square Test Model 10-CV"]).transpose()

#Residual Plot
fig, ax = plt.subplots()
plt.scatter(Y_pred_adelaide,(y-Y_pred_adelaide))
plt.axhline(y=0, color='r', linestyle='-')
pl.title("Residual Plot : Model Adelaide")
pl.xlabel("Predicted Value")
pl.ylabel("Residual")
fig.savefig("Residual Plot Adelaide Model.png")

#brisbane data
#select temperature, humidity, wind speed and pressure features
weather_brisbane_selected=weather_brisbane.loc[:, weather_brisbane.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','3pm wind speed (km/h)','9am wind speed (km/h)'])]
weather_brisbane_selected=weather_brisbane_selected.set_index('Date')

#transform wind speed value
weather_brisbane_selected.loc[weather_brisbane_selected['3pm wind speed (km/h)'] == "Calm",'3pm wind speed (km/h)'] = 1
weather_brisbane_selected.loc[weather_brisbane_selected['9am wind speed (km/h)'] == "Calm",'9am wind speed (km/h)'] = 1
weather_brisbane_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_brisbane_selected['3pm wind speed (km/h)'], errors='ignore')
weather_brisbane_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_brisbane_selected['9am wind speed (km/h)'], errors='ignore')

#exclude outlier
standard_deviations = 3
weather_brisbane_nonoutlier=weather_brisbane_selected[weather_brisbane_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_brisbane_nona=weather_brisbane_nonoutlier.dropna()

#merging total demand to weather condition
brisbane_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "QLD1"].merge(weather_brisbane_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
brisbane_corelation = brisbane_merge_data.corr()
brisbane_corelation.to_excel("brisbane_corelation_features.xlsx")

#exclude collinerity in independent feature
brisbane_corelation_selected=brisbane_corelation.loc[brisbane_corelation.index.isin(brisbane_corelation.iloc[0][(brisbane_corelation.iloc[0].abs() > 0.1) ].index),brisbane_corelation.iloc[0][(brisbane_corelation.iloc[0].abs() > 0.1) ].index]
brisbane_corelation_selected=brisbane_corelation_selected.reindex(brisbane_corelation_selected.TOTALDEMAND.abs().sort_values(ascending = False).index)
selected_feauture  = []
for i in range(1,len(brisbane_corelation_selected)) :
  checked_feature = brisbane_corelation_selected.index[i]
  corr_checked = brisbane_corelation_selected.iloc[i]
  if i == 1 :
    selected_feauture.append(checked_feature)
  else :
    if (corr_checked[corr_checked.index.isin(brisbane_corelation_selected.index[1:i])].abs() > 0.5).any() :
      continue
    else :
      selected_feauture.append(checked_feature)

#regression
X=brisbane_merge_data[selected_feauture].reset_index().drop(['index'], axis=1)
y=brisbane_merge_data["TOTALDEMAND"].reset_index().drop(['index'], axis=1)
p = np.array([1, 2])
reg = LinearRegression().fit(X, y)
r_square_brisbane=reg.score(X, y)
coef_regression_brisbane = reg.coef_
intercept_regression_brisbane = reg.intercept_
Y_pred_brisbane = reg.predict(X)
MSE_brisbane = mean_squared_error(y,Y_pred_brisbane)
brisbane_coef = pd.DataFrame([["intercept"]+selected_feauture,intercept_regression_brisbane.tolist() + reg.coef_[0].tolist()],["cofficient name","cofficient value"]).transpose()
brisbane_coef.to_excel("brisbane_model_coefficient.xlsx")

#creating experiment design
#set k=10 value for k-fold CV
k = 10

kf_CV = KFold(n_splits=k, shuffle=True, random_state=42)
results_MSE_brisbane = []
results_r_square_train_brisbane = []
results_r_square_test_brisbane = []

for train_idx, test_idx in kf_CV.split(X):
    # train-test split
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        
    # Training
    reg = LinearRegression().fit(X_train, y_train)
    r_square_train=reg.score(X_train, y_train) 
    r_square_test=reg.score(X_test, y_test)   
    
    # Predictions
    Y_pred = reg.predict(X_test)
    MSE = mean_squared_error(y_test,Y_pred)
    results_MSE_brisbane.append(MSE)
    results_r_square_train_brisbane.append(r_square_train)
    results_r_square_test_brisbane.append(r_square_test)

#making performence metric
performence_metric_brisbane = pd.DataFrame([MSE_brisbane,r_square_brisbane,st.mean(results_MSE_brisbane),st.mean(results_r_square_train_brisbane),st.mean(results_r_square_test_brisbane)],["MSE Original Model","R Square Original Model","MSE 10-CV","R Square Train Model 10-CV","R Square Test Model 10-CV"]).transpose()

#Residual Plot
fig, ax = plt.subplots()
plt.scatter(Y_pred_brisbane,(y-Y_pred_brisbane))
plt.axhline(y=0, color='r', linestyle='-')
pl.title("Residual Plot : Model Brisbane")
pl.xlabel("Predicted Value")
pl.ylabel("Residual")
fig.savefig("Residual Plot Brisbane Model.png")

#melbourne data
##select temperature, humidity, wind speed and pressure features
weather_melbourne_selected=weather_melbourne.loc[:, weather_melbourne.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','3pm wind speed (km/h)','9am wind speed (km/h)'])]
weather_melbourne_selected=weather_melbourne_selected.set_index('Date')

#transform wind speed value
weather_melbourne_selected.loc[weather_melbourne_selected['3pm wind speed (km/h)'] == "Calm",'3pm wind speed (km/h)'] = 1
weather_melbourne_selected.loc[weather_melbourne_selected['9am wind speed (km/h)'] == "Calm",'9am wind speed (km/h)'] = 1
weather_melbourne_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_melbourne_selected['3pm wind speed (km/h)'], errors='ignore')
weather_melbourne_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_melbourne_selected['9am wind speed (km/h)'], errors='ignore')

#exclude outlier
standard_deviations = 3
weather_melbourne_nonoutlier=weather_melbourne_selected[weather_melbourne_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_melbourne_nona=weather_melbourne_nonoutlier.dropna()

#merging total demand to weather condition
melbourne_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "VIC1"].merge(weather_melbourne_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
melbourne_corelation = melbourne_merge_data.corr()
melbourne_corelation.to_excel("melbourne_corelation_features.xlsx")

#exclude collinerity in independent feature
melbourne_corelation_selected=melbourne_corelation.loc[melbourne_corelation.index.isin(melbourne_corelation.iloc[0][(melbourne_corelation.iloc[0].abs() > 0.1) ].index),melbourne_corelation.iloc[0][(melbourne_corelation.iloc[0].abs() > 0.1) ].index]
melbourne_corelation_selected=melbourne_corelation_selected.reindex(melbourne_corelation_selected.TOTALDEMAND.abs().sort_values(ascending = False).index)
selected_feauture  = []
for i in range(1,len(melbourne_corelation_selected)) :
  checked_feature = melbourne_corelation_selected.index[i]
  corr_checked = melbourne_corelation_selected.iloc[i]
  if i == 1 :
    selected_feauture.append(checked_feature)
  else :
    if (corr_checked[corr_checked.index.isin(melbourne_corelation_selected.index[1:i])].abs() > 0.5).any() :
      continue
    else :
      selected_feauture.append(checked_feature)

#regression
X=melbourne_merge_data[selected_feauture].reset_index().drop(['index'], axis=1)
y=melbourne_merge_data["TOTALDEMAND"].reset_index().drop(['index'], axis=1)
reg = LinearRegression().fit(X, y)
r_square_melbourne=reg.score(X, y)
coef_regression_melbourne = reg.coef_
intercept_regression_melbourne = reg.intercept_
Y_pred_melbourne = reg.predict(X)
MSE_melbourne = mean_squared_error(y,Y_pred_melbourne)
melbourne_coef = pd.DataFrame([["intercept"]+selected_feauture,intercept_regression_melbourne.tolist() + reg.coef_[0].tolist()],["cofficient name","cofficient value"]).transpose()
melbourne_coef.to_excel("melbourne_model_coefficient.xlsx")

#creating experiment design
# set k=10 value for k-fold CV
k = 10

kf_CV = KFold(n_splits=k, shuffle=True, random_state=42)
results_MSE_melbourne = []
results_r_square_train_melbourne = []
results_r_square_test_melbourne = []

for train_idx, test_idx in kf_CV.split(X):
    # train-test split
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        
    # Training
    reg = LinearRegression().fit(X_train, y_train)
    r_square_train=reg.score(X_train, y_train) 
    r_square_test=reg.score(X_test, y_test)   
    
    # Predictions
    Y_pred = reg.predict(X_test)
    MSE = mean_squared_error(y_test,Y_pred)
    results_MSE_melbourne.append(MSE)
    results_r_square_train_melbourne.append(r_square_train)
    results_r_square_test_melbourne.append(r_square_test)

#making performence metric
performence_metric_melbourne = pd.DataFrame([MSE_melbourne,r_square_melbourne,st.mean(results_MSE_melbourne),st.mean(results_r_square_train_melbourne),st.mean(results_r_square_test_melbourne)],["MSE Original Model","R Square Original Model","MSE 10-CV","R Square Train Model 10-CV","R Square Test Model 10-CV"]).transpose()

#Residual Plot
fig, ax = plt.subplots()
plt.scatter(Y_pred_melbourne,(y-Y_pred_melbourne))
plt.axhline(y=0, color='r', linestyle='-')
pl.title("Residual Plot : Model Melbourne")
pl.xlabel("Predicted Value")
pl.ylabel("Residual")
fig.savefig("Residual Plot Melbourne Model.png")

#sydney data
#select temperature, humidity, wind speed and pressure features
weather_sydney_selected=weather_sydney.loc[:, weather_sydney.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','3pm wind speed (km/h)','9am wind speed (km/h)'])]
weather_sydney_selected=weather_sydney_selected.set_index('Date')

#transform wind speed value
weather_sydney_selected.loc[weather_sydney_selected['3pm wind speed (km/h)'] == "Calm",'3pm wind speed (km/h)'] = 1
weather_sydney_selected.loc[weather_sydney_selected['9am wind speed (km/h)'] == "Calm",'9am wind speed (km/h)'] = 1
weather_sydney_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_sydney_selected['3pm wind speed (km/h)'], errors='ignore')
weather_sydney_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_sydney_selected['9am wind speed (km/h)'], errors='ignore')

#exclude outlier
standard_deviations = 3
weather_sydney_nonoutlier=weather_sydney_selected[weather_sydney_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_sydney_nona=weather_sydney_nonoutlier.dropna()

#merging total demand to weather condition
sydney_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "VIC1"].merge(weather_sydney_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
sydney_corelation = sydney_merge_data.corr()
sydney_corelation.to_excel("sydney_corelation_features.xlsx")

#exclude collinerity in independent feature
sydney_corelation_selected=sydney_corelation.loc[sydney_corelation.index.isin(sydney_corelation.iloc[0][(sydney_corelation.iloc[0].abs() > 0.1) ].index),sydney_corelation.iloc[0][(sydney_corelation.iloc[0].abs() > 0.1) ].index]
sydney_corelation_selected=sydney_corelation_selected.reindex(sydney_corelation_selected.TOTALDEMAND.abs().sort_values(ascending = False).index)
selected_feauture  = []
for i in range(1,len(sydney_corelation_selected)) :
  checked_feature = sydney_corelation_selected.index[i]
  corr_checked = sydney_corelation_selected.iloc[i]
  if i == 1 :
    selected_feauture.append(checked_feature)
  else :
    if (corr_checked[corr_checked.index.isin(sydney_corelation_selected.index[1:i])].abs() > 0.5).any() :
      continue
    else :
      selected_feauture.append(checked_feature)

#regression
X=sydney_merge_data[selected_feauture].reset_index().drop(['index'], axis=1)
y=sydney_merge_data["TOTALDEMAND"].reset_index().drop(['index'], axis=1)
reg = LinearRegression().fit(X, y)
r_square_sydney=reg.score(X, y)
coef_regression_sydney = reg.coef_
intercept_regression_sydney = reg.intercept_
Y_pred_sydney = reg.predict(X)
MSE_sydney = mean_squared_error(y,Y_pred_sydney)
sydney_coef = pd.DataFrame([["intercept"]+selected_feauture,intercept_regression_sydney.tolist() + reg.coef_[0].tolist()],["cofficient name","cofficient value"]).transpose()
sydney_coef.to_excel("sydney_model_coefficient.xlsx")

#creating experiment design
# set k=10 value for k-fold CV
k = 10

kf_CV = KFold(n_splits=k, shuffle=True, random_state=42)
results_MSE_sydney = []
results_r_square_train_sydney = []
results_r_square_test_sydney = []

for train_idx, test_idx in kf_CV.split(X):
    # train-test split
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        
    # Training
    reg = LinearRegression().fit(X_train, y_train)
    r_square_train=reg.score(X_train, y_train) 
    r_square_test=reg.score(X_test, y_test)   
    
    # Predictions
    Y_pred = reg.predict(X_test)
    MSE = mean_squared_error(y_test,Y_pred)
    results_MSE_sydney.append(MSE)
    results_r_square_train_sydney.append(r_square_train)
    results_r_square_test_sydney.append(r_square_test)

#making performence metric
performence_metric_sydney = pd.DataFrame([MSE_sydney,r_square_sydney,st.mean(results_MSE_sydney),st.mean(results_r_square_train_sydney),st.mean(results_r_square_test_sydney)],["MSE Original Model","R Square Original Model","MSE 10-CV","R Square Train Model 10-CV","R Square Test Model 10-CV"]).transpose()

#Residual Plot
fig, ax = plt.subplots()
plt.scatter(Y_pred_sydney,(y-Y_pred_sydney))
plt.axhline(y=0, color='r', linestyle='-')
pl.title("Residual Plot : Model Sydney")
pl.xlabel("Predicted Value")
pl.ylabel("Residual")
fig.savefig("Residual Plot Sydney Model.png")

#combine performence matrix for all cities
performence_metrix_all = pd.concat([performence_metric_adelaide,performence_metric_brisbane,performence_metric_melbourne,performence_metric_sydney])
performence_metrix_all.index = ["Adelaide model","Brisbane model","Melbourne model","Sydney model"]
performence_metrix_all.to_excel("performence_metrix.xlsx")
