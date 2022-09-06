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

restaurants['real_id'] = find_partitions(
    df=restaurants,
    match_func=same_restaurant
)
restaurants
restaurants['real_id'] = find_partitions(
    df=restaurants,
    match_func=same_restaurant,
    max_size=2
)

restaurants

df=restaurants[['name','real_id']].drop_duplicates()
df
#restaurants['area_code'] = restaurants['phone'].str.split(' ', expand=True)[0]

#restaurants['real_id'] = find_partitions(
    df=restaurants,
    match_func=same_restaurant,
    block_by='area_code',
    max_size=30
)

#restaurants[restaurants['area_code']=='NaN']
