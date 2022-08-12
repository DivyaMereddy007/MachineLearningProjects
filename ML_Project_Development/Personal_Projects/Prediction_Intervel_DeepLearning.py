
# load and summarize the housing dataset
from pandas import read_csv
from matplotlib import pyplot
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
# summarize first few lines
print(dataframe.head())

# split into input and output values
X, y = values[:,:-1], values[:,-1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)

# scale input data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# load and prepare the dataset for modeling
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
values = dataframe.values
# split into input and output values
X, y = values[:,:-1], values[:,-1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)
# scale input data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# summarize
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
