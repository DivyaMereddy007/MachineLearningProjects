import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
try:
    import theano
except:
    !pip install Theano
import theano
#import tensorflow
#from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os

churn_data = pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Fuzzylogic/Submit_folder/css_public_all_ofos_locations.csv', on_bad_lines='skip',sep='\')


len(churn_data)

churn_data.head()
churn_data.info()

churn_data.describe()

churn_data.head()

churn_data=churn_data[~churn_data['name'].isna() ]
churn_data=churn_data[~churn_data['name'].str.contains("\[]",na=False)]
churn_data=churn_data[churn_data['active']=='TRUE']
churn_data.count()
churn_data['restaurant_chain'][((churn_data['restaurant_chain'].isna()==True) | (churn_data['restaurant_chain']=='TRUE')| ((churn_data['restaurant_chain'].str.contains(':')==True)) | (churn_data['restaurant_chain'].str.isnumeric()))]=None

churn_data['restaurant_chain'][((churn_data['restaurant_chain'].isna()!=True) | (churn_data['restaurant_chain']=='TRUE')| ((churn_data['restaurant_chain'].str.contains(':')==True)))]


churn_data.groupby(['restaurant_chain'])['restaurant_id'].count()

df=churn_data[['name','latitude','longitude',
'city','country','active','standardized_name',
'delivery_radius'
]]
df.describe()


churn_data.groupby(['standardized_name'])['standardized_name'].count()

churn_data['address']=churn_data['city']+' '+churn_data['country']

churn_data['country'][((churn_data['country'].isna()==True) | (churn_data['country']=='TRUE')| ((churn_data['country'].str.contains(':')==True)))]=None


churn_data.groupby(['country'])['country'].count()

churn_data['city'][((churn_data['city'].isna()!=True) &( (churn_data['city']=='TRUE')| ((churn_data['city'].str.contains(':')==True))))] =np.nan

churn_data.groupby(['city'])['city'].count()

churn_data.groupby(['address'])['address'].count()

churn_data.groupby(['latitude','longitude','name']).count()

churn_data['standardized_name'][churn_data['standardized_name'].str.contains('pizza')]



####++++++++++++++++++++++++++++++++++++++++++####++++++++++++++++++++++++++++++++++++++++++####++++++++++++++++++++++++++++++++++++++++++####++++++++++++++++++++++++++++++++++++++++++####++++++++++++++++++++++++++++++++++++++++++####++++++++++++++++++++++++++++++++++++++++++

churn_data.drop(['CustomerId','Surname'],axis=1,inplace=True)
# some columns have text data so let's one hot encode them
#  for more on one hot encoding click this link below
# https://www.kaggle.com/shrutimechlearn/types-of-regression-and-stats-in-depth
Geography_dummies = pd.get_dummies(prefix='Geo',data=churn_data,columns=['Geography'])


Geography_dummies.head()

Gender_dummies = Geography_dummies.replace(to_replace={'Gender': {'Female': 1,'Male':0}})

Gender_dummies.head()

churn_data_encoded = Gender_dummies

sns.countplot(y=churn_data_encoded.Exited ,data=churn_data_encoded)
plt.xlabel("Count of each Target class")
plt.ylabel("Target classes")
plt.show()


churn_data_encoded.hist(figsize=(15,12),bins = 15)
plt.title("Features Distribution")
plt.show()


plt.figure(figsize=(15,15))
p=sns.heatmap(churn_data_encoded.corr(), annot=True,cmap='RdYlGn',center=0)


fig,ax = plt.subplots(nrows = 4, ncols=3, figsize=(30,30))
row = 0
col = 0
for i in range(len(churn_data_encoded.columns) -1):
    if col > 2:
        row += 1
        col = 0
    axes = ax[row,col]
    sns.boxplot(x = churn_data_encoded['Exited'], y = churn_data_encoded[churn_data_encoded.columns[i]],ax = axes)
    col += 1
plt.tight_layout()
# plt.title("Individual Features by Class")
plt.show()
