#link
import pandas as pd
import numpy as np
#input data
data = pd.read_csv('MachineLearningProjects/ML_Project_Development/Recommendation_Systems_Collaborative_Filtering_Co-Occurrence/data.csv')
data.head()
data.set_index('Customer', inplace = True)
data
# calculate co-occurrence matrix
x = np.array(data)
y = np.array(data.T)
co_matrix = np.dot(y,x)
np.fill_diagonal(co_matrix, 0)
df_co = pd.DataFrame(co_matrix, columns = ['Apple', 'Banana', 'Carrot', 'Milk'], index = ['Apple', 'Banana', 'Carrot', 'Milk'])
df_co

# calculate user recommendation matrix
user_matrix = np.dot(x, co_matrix)
idx = pd.np.nonzero(x)
user_matrix[idx] = 0
df_user_recommend = pd.DataFrame(user_matrix, columns = ['Apple', 'Banana', 'Carrot', 'Milk'], index = ['Ann', 'Bob', 'John', 'Jane', 'Susan', 'Tod', 'Tom', 'Willam', 'Zac'])

df_user_recommend

#Python Code for Amazon Personalize



import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
data = pd.read_csv('data1.csv')

user_category = CategoricalDtype(sorted(data.Customer.unique()), ordered=True)
item_category = CategoricalDtype(sorted(data.Item.unique()), ordered=True)

row = data['Customer'].astype(user_category).cat.codes
col = data['Item'].astype(item_category).cat.codes

data['Count'] = 1

sparse_matrix = csr_matrix((data['Count'], (row, col)),shape=(user_category.categories.size,item_category.categories.size))
sparse_df = pd.SparseDataFrame(sparse_matrix,index=user_category.categories,columns=item_category.categories,default_fill_value=0)

co_matrix = sparse_matrix.transpose().dot(sparse_matrix)
co_matrix.setdiag(0)
co_df = pd.SparseDataFrame(co_matrix,
                               index=item_category.categories,
                               columns=item_category.categories,
                               default_fill_value=0)

idx = pd.np.nonzero(co_matrix)

rows = idx[0]
columns = idx[1]
subjects = [item_id_category.categories[i] for i in rows]
peers = [item_id_category.categories[i] for i in columns]

ele = co_matrix[idx].tolist()[0]

df_recommend = pd.DataFrame.from_records(zip(subs, peers, ele), columns = ['subject', 'peers', 'peer_counts'])


#developing co occurance model to other data set.


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
import pandas as pd
import numpy as np
ratings_train = pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Recommendation_systems_User_Interest_Prediction/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Recommendation_systems_User_Interest_Prediction/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape
ratings_train['recommend']=0
ratings_train['recommend'][ratings_train['rating']>2]=1

ratings_test['recommend']=0
ratings_test['recommend'][ratings_test['rating']>2]=1

ratings_train.head()
ratings_test.head()
ratings=ratings_train
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

user_category = sorted(ratings_train.user_id.unique())
item_category = sorted(ratings_train.movie_id.unique())

row =  ratings.user_id
col = ratings.movie_id
ratings_train.shape
ratings_train['Count']=1
sparse_matrix = csr_matrix((ratings_train['Count'], (row, col)),shape=(n_users,n_items))
sparse_df = pd.SparseDataFrame(sparse_matrix,index=user_category.categories,columns=item_category.categories,default_fill_value=0)
