#semantic similarity

from io import StringIO
s = StringIO("""abby_john         abc   abc   abc   abc
abby_johnny       def   def   def   def
a_j               ghi   ghi   ghi   ghi
abby_(john)       abc   abc   abc   abc
abby_john_doe     def   def   def   def
aby_John_Doedy    ghi   ghi   ghi   ghi
abby john         ghi   ghi   ghi   ghi
john_abby_doe     def   def   def   def
aby_/_John_Doedy  ghi   ghi   ghi   ghi
doe jane          abc   abc   abc   abc
doe_jane          def   def   def   def""")

import pandas as pd
df = pd.read_fwf(s,header=None,sep='\s+')
lst_original = df[0].tolist() # the first column

#Vectorize (turn into numerical representation):

import numpy as np
from gensim.models import Word2Vec

m = Word2Vec(lst_original,min_count=1,cbow_mean=1)
def vectorizer(sent,m):
    vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else:
                vec = np.add(vec, m[w])
            numw += 1
        except Exception as e:
            print(e)
    return np.asarray(vec) / numw

l = []
for i in lst_original:
    l.append(vectorizer(i,m))

X = np.array(l)
X

from sklearn.cluster import KMeans

clf = KMeans(n_clusters=2,init='k-means++',n_init=100,random_state=0)
labels = clf.fit_predict(X)

previous_cluster = 0
for index, sentence in enumerate(lst_original):
    if index > 0:
        previous_cluster = labels[index - 1]
    cluster = labels[index]
    if previous_cluster != cluster:
        print(str(labels[index]) + ":" + str(sentence))

############################################################################################################################################################

import difflib
import re

def similarity_replace(series):

    reverse_map = {}
    diz_map = {}
    for i,s in series.iteritems():
        diz_map[s] = re.sub(r'[^a-z]', '', s.lower())
        reverse_map[re.sub(r'[^a-z]', '', s.lower())] = s

    best_match = {}
    uni = list(set(diz_map.values()))
    for w in uni:
        best_match[w] = sorted(difflib.get_close_matches(w, uni, n=3, cutoff=0.5), key=len)[0]

    return series.map(diz_map).map(best_match).map(reverse_map)

df = pd.DataFrame({'name':['abby_john','abby_johnny','a_j','abby_(john)','john_abby_doe','aby_/_John_Doedy'],
                       'col1':['abc','add','sda','sas','sad','ass'],
                       'col2':['abc','add','sda','sas','sad','ass'],
                       'col3':['abc','add','sda','sas','sad','ass']})

df['name'] = similarity_replace(df.name)
df

#--------

import difflib
import re

def similarity_replace(series):

    reverse_map = {}
    diz_map = {}
    for i,s in series.iteritems():
        diz_map[s] = re.sub(r'[^a-z]', '', s.lower())
        reverse_map[re.sub(r'[^a-z]', '', s.lower())] = s

    best_match = {}
    uni = list(set(diz_map.values()))
    for w in uni:
        best_match[w] = sorted(difflib.get_close_matches(w, uni, n=3, cutoff=0.5), key=len)[0]

    return series.map(diz_map).map(best_match).map(reverse_map)

df =  pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/Fuzzylogic/Submit_folder/css_public_all_ofos_locations.csv', on_bad_lines='skip',sep='\')

df.head()
len(df)
restaurants=df
restaurants['geom'][(restaurants['geom'].isna()!=True) ].values

df=df[['name',#'latitude','longitude',
'city','country','active','standardized_name',
#'delivery_radius'
]]
df=df[df['name'].isna()==False]

df=similarity_replace(df.name)
df
