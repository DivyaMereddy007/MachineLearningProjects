#https://maxhalford.github.io/blog/transitive-duplicates/
from fuzzywuzzy import fuzz

def is_same_user(user_1, user_2):
    return fuzz.partial_ratio(user_1['first_name'], user_2['first_name']) > 90


import numpy as np
import pandas as pd


def find_partitions(df, match_func, max_size=None, block_by=None):
    """Recursive algorithm for finding duplicates in a DataFrame."""

    # If block_by is provided, then we apply the algorithm to each block and
    # stitch the results back together
    if block_by is not None:
        blocks = df.groupby(block_by).apply(lambda g: find_partitions(
            df=g,
            match_func=match_func,
            max_size=max_size
        ))

        keys = blocks.index.unique(block_by)
        for a, b in zip(keys[:-1], keys[1:]):
            blocks.loc[b, :] += blocks.loc[a].iloc[-1] + 1

        return blocks.reset_index(block_by, drop=True)

    def get_record_index(r):
        return r[df.index.name or 'index']

    # Records are easier to work with than a DataFrame
    records = df.to_records()

    # This is where we store each partition
    partitions = []

    def find_partition(at=0, partition=None, indexes=None):

        r1 = records[at]

        if partition is None:
            partition = {get_record_index(r1)}
            indexes = [at]

        # Stop if enough duplicates have been found
        if max_size is not None and len(partition) == max_size:
            return partition, indexes

        for i, r2 in enumerate(records):

            if get_record_index(r2) in partition or i == at:
                continue

            if match_func(r1, r2):
                print('partition',partition)
                partition.add(get_record_index(r2))
                indexes.append(i)
                print('indexes',indexes)
                find_partition(at=i, partition=partition, indexes=indexes)

        return partition, indexes

    while len(records) > 0:
        partition, indexes = find_partition()
        partitions.append(partition)
        records = np.delete(records, indexes)

    return pd.Series({
        idx: partition_id
        for partition_id, idxs in enumerate(partitions)
        for idx in idxs
    })


from fuzzywuzzy import fuzz


def same_phone(r1, r2):
    return r1['phone'] == r2['phone']


def same_area_code(r1, r2):
    return r1['phone'].split(' ')[0] == r2['phone'].split(' ')[0]


def same_name(r1, r2):
    return fuzz.ratio(r1['name'], r2['name']) > 75


def similar_address(r1, r2):
    return (
        fuzz.ratio(r1['address'], r2['address']) > 55 or
        fuzz.partial_ratio(r1['address'], r2['address']) > 75
    )

def similar_name(r1, r2):
    return fuzz.partial_ratio(r1['name'], r2['name']) > 50


def manual_ritz(r1, r2):
    if 'ritz carlton' in r1['name']:
        for term in ['cafe', 'dining room', 'restaurant']:
            if term in r1['name']:
                return term in r2['name']
    return True


def manual_le_marais(r1, r2):
    return not (
        r1['name'] == 'le marais' and r2['name'] == 'le madri' or
        r1['name'] == 'le madri' and r2['name'] == 'le marais'
    )

df_2.head()
def same_restaurant(r1, r2):
    return (
        (
            (
                same_phone(r1, r2) and
                similar_name(r1, r2)
            ) or
            (
                same_area_code(r1, r2) and
                same_name(r1, r2) and
                similar_address(r1, r2)
            )
        ) and
        manual_ritz(r1, r2) and
        manual_le_marais(r1, r2)
    )

import pandas as pd

restaurants = pd.read_csv('MachineLearningProjects/ML_Project_Development/Fuzzylogic/Sample_Data/restaurants.tsv',sep='\t',index_col='id')

restaurants.head()

restaurants['area_code'] = restaurants['phone'].str.split(' ', expand=True)[0]
import numpy as np

restaurants['real_id'] = find_partitions(
    df=restaurants,
    match_func=same_restaurant
)
restaurants
#####understnaidng above function###########################################################################
df=restaurants
match_func=same_restaurant
max_size=None
block_by='area_code'


if block_by is not None:
    blocks = df.groupby(block_by).apply(lambda g: find_partitions(
        df=g,
        match_func=match_func,
        max_size=max_size
    ))

    keys = blocks.index.unique(block_by)
    for a, b in zip(keys[:-1], keys[1:]):
        blocks.loc[b, :] += blocks.loc[a].iloc[-1] + 1

    print(blocks.reset_index(block_by, drop=True))

blocks.index.unique(block_by)
def get_record_index(r):
    return r[df.index.name or 'index']

# Records are easier to work with than a DataFrame
records = df.to_records()

# This is where we store each partition
partitions = []
records[0]

records
r1 = records[0]
{get_record_index(r1)}

at=0; partition=None; indexes=None
r1 = records[at]
r1

for i, r2 in enumerate(records):

    if get_record_index(r2) in partition or i == at:
        continue

    if 1:
        partition.add(get_record_index(r2))
        indexes.append(i)
        print(get_record_index(r2),i)
        #find_partition(at=i, partition=partition, indexes=indexes)
match_func(r1, r2)

if partition is None:
    partition = {get_record_index(r1)}
    indexes = [at]

if max_size is not None and len(partition) == max_size:
    return partition, indexes

def find_partition(at=0, partition=None, indexes=None):

    r1 = records[at]

    if partition is None:
        partition = {get_record_index(r1)}
        indexes = [at]

    # Stop if enough duplicates have been found
    if max_size is not None and len(partition) == max_size:
        return partition, indexes

    for i, r2 in enumerate(records):

        if get_record_index(r2) in partition or i == at:
            continue

        if match_func(r1, r2):
            partition.add(get_record_index(r2))
            indexes.append(i)
            find_partition(at=i, partition=partition, indexes=indexes)

    return partition, indexes

while len(records) > 0:
    partition, indexes = find_partition()
    partitions.append(partition)
    records = np.delete(records, indexes)

return pd.Series({
    idx: partition_id
    for partition_id, idxs in enumerate(partitions)
    for idx in idxs
})


def find_partitions(df, match_func, max_size=None, block_by=None):
    """Recursive algorithm for finding duplicates in a DataFrame."""

df=restaurants[:10]
len(df)
df_3=df_3.sort_values(by=['address'])

records = df_2.to_records()
df_3['new_key_address']=-1
#df_3.iloc[0]['new_key_address']=0
#df_3.iloc[0]['new_key_address']

import pandas as pd
import random
# read the data from the downloaded CSV file.
data = pd.read_csv('https://s3-eu-west-1.amazonaws.com/shanebucket/downloads/uk-500.csv')
# set a numeric id for use as an index for examples.
data['id'] = [random.randint(0,1000) for x in range(data.shape[0])]
data.head(5)

data.iloc[0,0]=1

df_3.iloc[i+1,7]
df_3.iloc[:,8][df_3.iloc[:,7]==759]

len(df_2.new_key.unique())
len(df_2.columns)
df_3.iloc[0,8]=df_3.iloc[0,7]
for i in range(1,len(df_3)-1):
    #df_2.iloc[i] == df_2.iloc[i+1]
    print(i,'..............................................')
    if (df_3.iloc[i+1,7]!=df_3.iloc[i,7]) and (match_func(df_3.iloc[i], df_3.iloc[i+1])):
        print('match_fun found for ',df_3.iloc[i],'@@@@', df_3.iloc[i+1])
        df_3.iloc[i+1,8]=df_3.iloc[i,8]
        df_3.iloc[:,8][df_3.iloc[:,7]==df_3.iloc[i+1,7]]=df_3.iloc[i,8]
    else:
        df_3.iloc[i+1,8]=df_3.iloc[i,8]+1

len(df_3)
df_3['new_key_address'][df_3['new_key_address']==-1]=df_3['new_key']

len(df_3['new_key_address'].unique())
df_3=df_2[1:10]
for i in df_2[1:10]:
    print(i)
####################################################################################################
restaurants['real_id'] = find_partitions(
    df=restaurants,
    match_func=same_restaurant,
    max_size=2
)

restaurants

df=restaurants[['name','real_id']].drop_duplicates()
df

restaurants['real_id'] = find_partitions(
    df=restaurants,
    match_func=same_restaurant,
    block_by='area_code',
    max_size=30
)

#restaurants[restaurants['area_code']=='NaN']
