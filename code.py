#import package
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
price_demand = pd.read_csv("price_demand_data.csv")
weather_adelaide = pd.read_csv("weather_adelaide.csv")
weather_brisbane = pd.read_csv("weather_brisbane.csv")
weather_melbourne = pd.read_csv("weather_melbourne.csv")
weather_sydney = pd.read_csv("weather_sydney.csv")
#price demand transformation to average by date
price_demand['date']=price_demand['SETTLEMENTDATE'].str.split(" ",expand=True)[0].str.replace(r'/', '-', regex=True)
price_demand_selected = price_demand.loc[:,['REGION','TOTALDEMAND','date']]
price_demand_bydate = price_demand_selected.groupby(['date','REGION']).agg({'TOTALDEMAND':'mean'}).reset_index().set_index("date")

#adelaide data
#Replacing calm with the value 1 in "9am wind speed (km/h)" and 3pm wind speed (km/h)
weather_adelaide.loc[weather_adelaide["9am wind speed (km/h)"] == "Calm", "9am wind speed (km/h)"] = 1
weather_adelaide.loc[weather_adelaide["3pm wind speed (km/h)"] == "Calm", "3pm wind speed (km/h)"] = 1

#select temperature, humidity and pressure features
weather_adelaide_selected=weather_adelaide.loc[:, weather_adelaide.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','9am wind speed (km/h)','3pm wind speed (km/h)'])]
weather_adelaide_selected=weather_adelaide_selected.set_index('Date')
#weather_adelaide_selected.to_csv("weather_adelaide_selected.csv")
weather_adelaide_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_adelaide_selected['3pm wind speed (km/h)'], errors='ignore')
weather_adelaide_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_adelaide_selected['9am wind speed (km/h)'], errors='ignore')

#exclude outlier
standard_deviations = 3
weather_adelaide_nonoutlier=weather_adelaide_selected[weather_adelaide_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_adelaide_nona=weather_adelaide_nonoutlier.dropna()

#merging total demand to weather condition
adelaide_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "SA1"].merge(weather_adelaide_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
adelaide_corelation = adelaide_merge_data.corr().head(1)
#brisbane data
weather_brisbane.loc[weather_brisbane["9am wind speed (km/h)"] == "Calm", "9am wind speed (km/h)"] = 1
weather_brisbane.loc[weather_brisbane["3pm wind speed (km/h)"] == "Calm", "3pm wind speed (km/h)"] = 1
#select temperature, humidity and pressure features
weather_brisbane_selected=weather_brisbane.loc[:, weather_brisbane.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','9am wind speed (km/h)','3pm wind speed (km/h)'])]
weather_brisbane_selected=weather_brisbane_selected.set_index('Date')
weather_brisbane_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_brisbane_selected['3pm wind speed (km/h)'], errors='ignore')
weather_brisbane_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_brisbane_selected['9am wind speed (km/h)'], errors='ignore')
#exclude outlier
standard_deviations = 3
weather_brisbane_nonoutlier=weather_brisbane_selected[weather_brisbane_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_brisbane_nona=weather_brisbane_nonoutlier.dropna()

#merging total demand to weather condition
brisbane_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "QLD1"].merge(weather_brisbane_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
brisbane_corelation = brisbane_merge_data.corr().head(1)
#melbourne data
weather_melbourne.loc[weather_melbourne["9am wind speed (km/h)"] == "Calm", "9am wind speed (km/h)"] = 1
weather_melbourne.loc[weather_melbourne["3pm wind speed (km/h)"] == "Calm", "3pm wind speed (km/h)"] = 1
#select temperature, humidity and pressure features
weather_melbourne_selected=weather_melbourne.loc[:, weather_melbourne.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','9am wind speed (km/h)','3pm wind speed (km/h)'])]
weather_melbourne_selected=weather_melbourne_selected.set_index('Date')
weather_melbourne_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_melbourne_selected['3pm wind speed (km/h)'], errors='ignore')
weather_melbourne_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_melbourne_selected['9am wind speed (km/h)'], errors='ignore')
#exclude outlier
standard_deviations = 3
weather_melbourne_nonoutlier=weather_melbourne_selected[weather_melbourne_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_melbourne_nona=weather_melbourne_nonoutlier.dropna()

#merging total demand to weather condition
melbourne_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "VIC1"].merge(weather_melbourne_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
melbourne_corelation = melbourne_merge_data.corr().head(1)
#sydney data
weather_sydney.loc[weather_sydney["9am wind speed (km/h)"] == "Calm", "9am wind speed (km/h)"] = 1
weather_sydney.loc[weather_sydney["3pm wind speed (km/h)"] == "Calm", "3pm wind speed (km/h)"] = 1
#select temperature, humidity and pressure features
weather_sydney_selected=weather_sydney.loc[:, weather_sydney.columns.isin(['Date','Minimum temperature (°C)','Maximum temperature (°C)','9am Temperature (°C)','9am relative humidity (%)','9am MSL pressure (hPa)','3pm Temperature (°C)','3pm relative humidity (%)','3pm MSL pressure (hPa)','9am wind speed (km/h)','3pm wind speed (km/h)'])]
weather_sydney_selected=weather_sydney_selected.set_index('Date')
weather_sydney_selected['3pm wind speed (km/h)'] = pd.to_numeric(weather_sydney_selected['3pm wind speed (km/h)'], errors='ignore')
weather_sydney_selected['9am wind speed (km/h)'] = pd.to_numeric(weather_sydney_selected['9am wind speed (km/h)'], errors='ignore')
#exclude outlier
standard_deviations = 3
weather_sydney_nonoutlier=weather_sydney_selected[weather_sydney_selected.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]
weather_sydney_nona=weather_sydney_nonoutlier.dropna()

#merging total demand to weather condition
sydney_merge_data = price_demand_bydate[price_demand_bydate['REGION'] == "NSW1"].merge(weather_sydney_nona, left_index=True, right_index=True).drop('REGION', axis=1)

#calculate the correlation
sydney_corelation = sydney_merge_data.corr().head(1)
Corr_table = pd.concat([sydney_corelation, melbourne_corelation,brisbane_corelation,adelaide_corelation], axis=0)
Corr_table.to_csv("Corr_table.csv")
print(Corr_table.head())
# add wind speed and cloud
# "9am wind speed (km/h)" and 3pm wind speed (km/h)
