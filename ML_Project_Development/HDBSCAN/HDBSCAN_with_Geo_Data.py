import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import seaborn as sns
import sklearn.datasets as data
%matplotlib inline
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}



df = pd.read_csv('/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/HDBSCAN/data/summer-travel-gps-full.csv', encoding='utf-8')
df.head()

coords = df[['lat', 'lon']].to_numpy()
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))

print('HDBSCAN Work Started'.center(100, '-'))
#HDBSCAN

plt.scatter(coords.T[0], coords.T[1], color='b', **plot_kwds)

import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True,metric='haversine')
clusterer.fit(coords)


clusterer=hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=True, leaf_size=400,
    metric='euclidean', min_cluster_size=100, min_samples=None, p=None)


clusterer.fit(coords)

clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)


clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)


clusterer.condensed_tree_.plot()

 clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())


palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(coords.T[0], coords.T[1], c=cluster_colors, **plot_kwds)
