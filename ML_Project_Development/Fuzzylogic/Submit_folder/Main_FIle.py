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
               #@print('r1r2----',r1,r2)
               #@print('partition',partition)
                partition.add(get_record_index(r2))
                indexes.append(i)
               #@print('indexes',indexes)
                find_partition(at=i, partition=partition, indexes=indexes)

        return partition, indexes

    while len(records) > 0:
        partition, indexes = find_partition()
        partitions.append(partition)
        records = np.delete(records, indexes)
       #@print('#############',indexes)
       #@print('#############',records)

   #@print('@@@@@@@@@@@@@@@@@@@Final',partition,indexes)
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

def same_Geo(r1, r2):
    #print('same_Geo', r1['latitude'], r2['latitude'],r1['longitude'] , r2['longitude'])
    return ((r1['latitude'] == r2['latitude']) and (r1['longitude'] == r2['longitude']))

def similar_Geo(r1, r2):
    #print('similar_Geo', ((fuzz.ratio(r1['latitude'], r2['latitude']) )),( (fuzz.ratio(r1['longitude'], r2['longitude']))))
    #return ((r1['latitude'] == r2['latitude']) and (r1['longitude'] == r2['longitude']))
    return ((fuzz.ratio(r1['latitude'], r2['latitude']) > 75) and (fuzz.ratio(r1['longitude'], r2['longitude']) > 75))

def same_name(r1, r2):
    return fuzz.ratio(r1['name'], r2['name']) > 85
import math

def similar_address(r1, r2):
    print(r1['city'],r2['city'])
    return (
        #fuzz.ratio(r1['address'], r2['address']) > 55 or
        #fuzz.partial_ratio(r1['address'], r2['address']) > 75

        False if ~(isinstance(r1['city'], str) and isinstance(r1['city'], str))  #math.isnan(r2['city']))
        else fuzz.ratio(r1['city'], r2['city']) > 75
    )

def similar_name(r1, r2):
    #print('similar_name', fuzz.partial_ratio(r1['name'], r2['name']) > 50)
    return fuzz.partial_ratio(r1['name'], r2['name']) > 50

def similar_standardized_name(r1, r2):
    #print('similar_standardized_name', r1['standardized_name'], r2['standardized_name'])
    return fuzz.partial_ratio(r1['standardized_name'], r2['standardized_name']) > 50

def same_standardized_name(r1, r2):
    #print('same_standardized_name', r1['standardized_name'], r2['standardized_name'])
    return fuzz.ratio(r1['standardized_name'], r2['standardized_name']) > 85


def manual_ritz(r1, r2):
    if 'ritz carlton' in r1['name']:
        for term in ['cafe','Cafe', 'dining room', 'restaurant','Restaurant','Eatery','eatery','Catering','catering']:
            if term in r1['name']:
                return term in r2['name']
    return True


def manual_le_marais(r1, r2):
    return not (
        r1['name'] == 'le marais' and r2['name'] == 'le madri' or
        r1['name'] == 'le madri' and r2['name'] == 'le marais'
    )


def same_restaurant(r1, r2):
    return (
        (

            # (
            #
            #    similar_standardized_name(r1, r2)
            #    and similar_Geo(r1, r2)
                #similar_address(r1, r2)
            # )
            # or
            (
                #same_phone(r1, r2) and
                same_Geo(r1, r2)

                and
                similar_name(r1, r2)
            )or
            (
                #same_phone(r1, r2) and
                same_standardized_name(r1, r2)
                and similar_Geo(r1, r2)
                #and
                #similar_name(r1, r2)
            )
             or
            (
            #     #same_area_code(r1, r2) and
                same_name(r1, r2)
                and
                similar_address(r1, r2)
            )


            #or
            #name	platform	sub_platform logic here
         ) and
        manual_ritz(r1, r2) and
        manual_le_marais(r1, r2)
    )

import pandas as pd

#restaurants = pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Fuzzylogic/Submit_folder/css_public_all_ofos_locations.csv', on_bad_lines='skip')
#len(restaurants)
#restaurants.head()
restaurants = pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Fuzzylogic/Submit_folder/css_public_all_ofos_locations.csv', on_bad_lines='skip',sep='\')
restaurants.head()
len(restaurants)
restaurants[restaurants['active']=='TRUE'].count()
#restaurants=restaurants[1:10000]

#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@



data_copy = pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Fuzzylogic/Submit_folder/css_public_all_ofos_locations.csv', on_bad_lines='skip',sep='\')

data_copy=data_copy[~data_copy['name'].isna() ]
data_copy=data_copy[~data_copy['name'].str.contains("\[]",na=False)]
data_copy=data_copy[data_copy['active']=='TRUE']
data_copy.count()
data_copy['restaurant_chain'][((data_copy['restaurant_chain'].isna()==True) | (data_copy['restaurant_chain']=='TRUE')| ((data_copy['restaurant_chain'].str.contains(':')==True)) | (data_copy['restaurant_chain'].str.isnumeric()))]=None

data_copy['restaurant_chain'][((data_copy['restaurant_chain'].isna()!=True) | (data_copy['restaurant_chain']=='TRUE')| ((data_copy['restaurant_chain'].str.contains(':')==True)))]


data_copy.groupby(['restaurant_chain'])['restaurant_id'].count()

data_copy=data_copy[['name','latitude','longitude',
'city','country','active','standardized_name',
'delivery_radius'
]]
data_copy.describe()


data_copy.groupby(['standardized_name'])['standardized_name'].count()

data_copy['address']=data_copy['city']+' '+data_copy['country']

data_copy['country'][((data_copy['country'].isna()==True) | (data_copy['country']=='TRUE')| ((data_copy['country'].str.contains(':')==True)))]=None

import numpy as np
data_copy.groupby(['country'])['country'].count()

data_copy['city'][((data_copy['city'].isna()!=True) &( (data_copy['city']=='TRUE')| ((data_copy['city'].str.contains(':')==True))))] =np.nan

data_copy.groupby(['city'])['city'].count()

data_copy.groupby(['address'])['address'].count()

data_copy.groupby(['latitude','longitude','name']).count()

#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@#@@@@@@@
restaurants=data_copy
#@restaurants['restaurant_chain'][(restaurants['restaurant_chain'].isna()!=True) & (restaurants['restaurant_chain']!='TRUE')& ((restaurants['restaurant_chain'].str.contains(':')==False))].values

restaurants['delivery_radius'].values
#restaurants['geom'][restaurants['geom'].isna()==False]

restaurants['country'].unique()

restaurants['address']=restaurants['city']+' '+restaurants['country']

#restaurant_id	name	platform	sub_platform	latitude	longitude	city	country	active	standardized_name	restaurant_chain	delivery_radius	geom
restaurants['city'][(restaurants['city'].isna()!=True) & (restaurants['city']!='TRUE')& ((restaurants['city'].str.contains(':')==False))].values

restaurants['address'][(restaurants['address'].isna()!=True) & ((restaurants['address'].str.contains(':|TRUE')==False))]=None
restaurants['address']=restaurants['address'].astype(str)
#fuzz.ratio(3.4, 4.5)
len(restaurants)
df2=restaurants[:100]

df2=df2
#[['name','latitude','longitude','standardized_name']]
df2.isnull().sum()
df2.isna().sum()
df2['country'].unique()

df2_city_NA=df2[df2['city'].isna()]
df2_city=df2[~df2['city'].isna()]

#tried to develop the system seperate for with city and with out city details.
# df2_city['real_id'] = find_partitions(
#     df=df2_city,
#     match_func=same_restaurant,
#     max_size=10,
#     block_by='city'
# )
#
# len(df2_city)
# df2_city['city'].unique()
#----
# df2_city_NA['real_id'] = find_partitions(
#     df=df2_city_NA,
#     match_func=same_restaurant,
#     max_size=10,
#     #block_by='city'
# )

df2['real_id'] = find_partitions(
    df=df2,
    match_func=same_restaurant,
    max_size=10,
    #block_by='city'
)
#tetsing purpose
#df2.iloc[0]
#df2.iloc[265]
#fuzz.partial_ratio('state_park', 'state_park_city')
#fuzz.ratio('state_park', 'palmyra')
#same_standardized_name(df2.iloc[11], df2.iloc[265])

len(df2)

#fuzz.partial_ratio('liberty_burger_-_keller', 'sgourmet_palace_china_bistro')
df2.head()
pd.set_option('display.max_rows', None)

#x=df2.groupby(by=['real_id']).agg({'name':'count'})
#x
df2.to_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Fuzzylogic/Submit_folder/Test/df2.csv')
#(df2[df2['real_id'].isin(x[x['name']>1].index.values)]).to_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Fuzzylogic/Submit_folder/Test/df2_agg.csv')
print(df2['real_id'].unique())
########################################################################################################################################################################################################################
#Logic to develop city and Non city data duplciates removal seperately
########################################################################################################################################################################################################################
# df2[df2['real_id']==7030]
#
#
# x=df2.groupby(by=['latitude','longitude']).agg({'name':'count'})
# df2[df2['real_id'].isin(x[x['name']>1].index.values)]
#
# #same_phone(r1, r2) and
# same_Geo(r1, r2) and
#  ((40.71418 == 34.164525) and (-74.015568 == -118.414288))
# fuzz.partial_ratio('Parm - Battery Park City', 'Gourmet Palace China Bistro')
#
# restaurants['real_id'].info()
#
# restaurants['real_id'] = find_partitions(
#     df=restaurants,
#     match_func=same_restaurant,
#     max_size=20
# )
#
# restaurants
#
# df=restaurants[['name','real_id']].drop_duplicates()
# df
# restaurants['area_code'] = restaurants['phone'].str.split(' ', expand=True)[0]
#
# # import sys
# # sys.setrecursionlimit(10000)
# restaurants['real_id'] = find_partitions(
#     df=restaurants,
#     match_func=same_restaurant,
#     block_by='address',
#     max_size=30
# )

#restaurants[restaurants['area_code']=='NaN']
