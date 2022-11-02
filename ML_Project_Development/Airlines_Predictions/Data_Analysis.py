from sklearn import metrics, linear_model

import pandas as pd
weather_data=pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Airlines_Predictions/Data/BainAirlineTaskDatasets/weather.csv')
#As I didnt find information about HIX fullform, I assumed HIX means Highland as there is only one other airport which doesnt seems lie HIX
weather_data=weather_data[(weather_data['airport']=='Highland')&(weather_data['time']>'2019-11-30')]

Flights_Data=pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Airlines_Predictions/Data/BainAirlineTaskDatasets/Flight_on_time_HIX.csv')
Flights_Data.columns
Flights_Data=Flights_Data[(Flights_Data['Airline']=='CA') & (Flights_Data['Origin_Airport']=='HIX')]
weather_data.head()

Flights_Data['Is_Delayed']=0
Flights_Data['Is_Delayed'][Flights_Data['Departure_Delay_Minutes']>15.0]=1
Flights_Data['Is_Delayed']
Flights_Data.head()

Flights_Data['Delay_Reason'].unique()

################################################################################################################################################################################################
################################################################################################################################################################################################
# as seen above there are mutiple reasons for the delay of the flights,
####Late Aircraft: it can be understood based on arrival wheels on Time
####Weather: It can be predicted with weather data.
####Carrier: Carrier delays are something that are created by Carrier cleaning services delay, waiting for avvrai of connecting flights delay, Its something we can improved
# so lets calculate the statistics for Carrier
#####Secuirty - we need to check it further
####AS NAS delays are created becuase of on-extreme weather conditions, airport operations, heavy traffic volume, air traffic control,
# we can try to check the statistics of the traphic of flights based on it, we can try to change the timings of the flights if we can
# We cannot influence the weather problems
################################################################################################################################################################################################
################################################################################################################################################################################################

Flights_Data.info()

Flights_Data.describe()



import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
sns.countplot(y=Flights_Data.Is_Delayed ,data=Flights_Data)
plt.xlabel("Count of each Target class")
plt.ylabel("Target classes")
plt.show()




##check here
weather_data.head()
weather_data['time'].unique()
weather_data['time']=pd.to_datetime(weather_data['time'])

weather_data['time'].unique()
Flights_Data['Scheduled_Departure_Time_In_Datetime']=pd.to_datetime(Flights_Data['FlightDate'])

#pd.to_timedelta(pd.to_datetime(Flights_Data['Scheduled_Departure_Time'].astype('int').apply(lambda x: '{0:0>4}'.format(x)), format='%H%M').dt.time)
#from datetime import date, time
Flights_Data['Time_Created']=pd.to_datetime(Flights_Data['Scheduled_Departure_Time'].astype('int').apply(lambda x: '{0:0>4}'.format(x)), format='%H%M').dt.time

Flights_Data['Scheduled_Departure_Datetime']=Flights_Data['Scheduled_Departure_Time_In_Datetime']+ pd.to_timedelta(Flights_Data['Time_Created'].astype(str))



Flights_Data.hist(figsize=(15,12),bins = 15)
plt.title("Features Distribution")
plt.show()

Flights_Data_ALS=Flights_Data.copy()

Flights_Data_ALS.drop(['Flight_Number'],axis=1,inplace=True)

plt.figure(figsize=(15,15))
p=sns.heatmap(Flights_Data_ALS.corr(), annot=True,cmap='RdYlGn',center=0)

plt.figure(figsize=(15,15))
p=sns.heatmap(Flights_Data_ALS[['Actual_Arrival_Time','Arrival_Delay_Minutes','Departure_Delay_Minutes']].corr(), annot=True,cmap='RdYlGn',center=0)


plt.figure(figsize=(15,15))
p=sns.heatmap(Flights_Data_ALS[['Actual_Arrival_Time','Arrival_Delay_Minutes','Departure_Delay_Minutes']][Flights_Data_ALS['Delay_Reason'].isin(['LateAircraft','Carrier'])].corr(), annot=True,cmap='RdYlGn',center=0)

################################################################################################################################################################################################
################################################################################################################################################################################################
#### Clearly departure time is depent on arrival time, and other arrival time related fetaures
####Late Aircraft seems like effected by  Late arrival slightly
#### even for the data excluding carrier delays, Security delays caused delayed data, still we saw similar correlation with Arrival_Delay_Minutes with Deoarture delay time
################################################################################################################################################################################################
################################################################################################################################################################################################


fig,ax = plt.subplots(nrows = 4, ncols=3, figsize=(30,30))
row = 0
col = 0
for i in range(len(Flights_Data.columns) -1):
    if col > 2:
        row += 1
        col = 0
    axes = ax[row,col]
    sns.boxplot(x = Flights_Data['Is_Delayed'], y = Flights_Data[Flights_Data.columns[i]],ax = axes)
    col += 1
plt.tight_layout()
# plt.title("Individual Features by Class")
plt.show()


######## Develop a Ml Systems #########

Flights_Data
##COmbine Weather and FLights Data

Flights_Data.columns

weather_data.columns

weather_data=weather_data.set_index('time')

Flights_Data['time']=Flights_Data['Scheduled_Departure_Datetime']
Flights_Data=Flights_Data.set_index('time')


weather_data.head()
Flights_Data.head()
Flights_Data.index.sort_values
weather_data.index.sort_values
Combine_Data=Flights_Data
Combine_Data[weather_data.columns]=weather_data
Combine_Data.head()


missing_df = Combine_Data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(Combine_Data.shape[0]-missing_df['missing values'])/Combine_Data.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)

#####Quality of the data should be improved either the data is missing or data need to be re evalualted

##Create categorical Variables
# Geography_dummies = pd.get_dummies(prefix='Geo',data=churn_data,columns=['Geography'])
#
#
# Geography_dummies.head()
#
# Gender_dummies = Geography_dummies.replace(to_replace={'Gender': {'Female': 1,'Male':0}})
#
# Gender_dummies.head()
#
# churn_data_encoded = Gender_dummies

def linear_regression(X,Y):
    #test = test2
    X = np.array(X)
    Y = np.array(Y)
    #X = X.reshape(len(X),1)
    #Y = Y.reshape(len(Y),1)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    result = regr.predict(X)
    return X, Y, result,regr

Combine_Data.columns
Y_col='Departure_Delay_Minutes'
X_col=['Arrival_Delay_Minutes','Arrival_Taxi', 'Arrival_WheelsOn','precipIntensity', 'precipProbability', 'temperature','apparentTemperature', 'dewPoint', 'humidity', 'pressure', 'windSpeed','windGust', 'windBearing', 'cloudCover', 'uvIndex', 'visibility','ozone', 'precipType', 'precipAccumulation']
import numpy as np
X=Combine_Data[X_col]
Y=Combine_Data[Y_col]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.3)
regr = linear_model.LinearRegression()

X, Y, result,model=linear_regression(X_train,y_train)

model
Y_Out=model.predict(X_test)

#Validation

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,Y_Out)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,Y_Out)

#R Square

model.score(X_train,y_train)


### Other variables also worth exploring further like validating whether time of data influence traphic or weather conditions or have any trend with carrier delays etc.
### We can validate date time with time series like XGBOOST


###########
#working on improving data. to match weather data and flighs dta use just DD/MM/YYYY HH not further
##
Flights_Data2=Flights_Data.reset_index()
weather_data2=weather_data.reset_index()
Flights_Data2['time']=
Combine_Data=Flights_Data
Combine_Data[weather_data.columns]=weather_data
Combine_Data.head()

pd.to_datetime(Flights_Data2['time'], format = '%Y-%m-%d %H')


## Further Need to correct the date colum and re run the code
