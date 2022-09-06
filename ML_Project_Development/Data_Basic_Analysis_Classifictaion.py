import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
try:
    import theano
except:
    !pip install Theano
import theano
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
print(os.listdir("/Users/divya.mereddy/Documents/GitHub/MachineLearningProjects/NeuralNetworks"))

# Importing the dataset
churn_data = pd.read_csv('/Users/divya.mereddy/Documents/GitHub/MachineLearningProjects/NeuralNetworks/Data/Churn_Modelling.csv',index_col='RowNumber')

churn_data.info()

churn_data.describe()

churn_data.head()

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
