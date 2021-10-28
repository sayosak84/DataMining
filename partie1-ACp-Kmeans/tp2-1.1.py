#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


def correlation_circle(df,nb_var,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j+2],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('fig/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)


# Q1

data = pd.read_csv('data/Data_World_Development_Indicators2.csv')

# print nb of instances and features
print(data.shape)   # 31 attributs

# print feature types
print(data.dtypes)  # les attributs  Country Name             object
					#				 Country Code             object 
                    # le reste des attributs sont des floats

# Q2

from sklearn.impute import SimpleImputer
dataNoMissingValues = SimpleImputer(strategy='median', missing_values=np.nan)
dataNoMissingValues = dataNoMissingValues.fit_transform(data[data.columns[2:]])  
print(dataNoMissingValues[0,3])   # verification d'une valeur vide = 76.8530589670305
print(dataNoMissingValues)

# Quel est l’intérêt d’utiliser ici la médiane plutôt que la moyenne ?
# La médiane est moins sensible aux valeurs extremes

print("###################################")
# Q3

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standarScaledData = standard_scaler.fit_transform(dataNoMissingValues)
print(standarScaledData)

# StandardScaler : centre les valeurs autour de 0 selon une échelle. C'est le résultat de x - Mean / ecartType

print("###################################")

#Q4

from sklearn.decomposition import PCA

acp = PCA(svd_solver='full')
coord = acp.fit_transform(standarScaledData)

# nb of computed components
print(acp.n_components_)

# explained variance scores
print(acp.explained_variance_ratio_)
# plot eigen values
n = np.size(standarScaledData, 0)
p = np.size(standarScaledData, 1)
eigval = float(n-1)/n*acp.explained_variance_
fig = plt.figure()
plt.plot(np.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.savefig('fig/acp_eigen_values')
plt.close(fig)


y = data["Country Code"]

# plot instances on the first plan (first 2 factors)
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-10,10)                 # on augmente les limites pour visualiser tout les pays
axes.set_ylim(-10,10)
for i in range(n):
    plt.annotate(y.values[i],(coord[i,0],coord[i,1]))
plt.plot([-10,10],[0,0],color='blue',linestyle='-',linewidth=1)
plt.plot([0,0],[-10,10],color='blue',linestyle='-',linewidth=1)
plt.title("ACP Factor 1 and 2")
plt.ylabel("Factor 1")
plt.xlabel("Factor 2")
plt.savefig('fig/acp_instances_1st_plan')
plt.close(fig)


# plot instances on the second plan (3rd and 4th factors)
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-10,10)
axes.set_ylim(-10,10)
for i in range(n):
    plt.annotate(y.values[i],(coord[i,2],coord[i,3]))
plt.plot([-10,10],[0,0],color='red',linestyle='-',linewidth=1)
plt.plot([0,0],[-10,10],color='red',linestyle='-',linewidth=1)
plt.savefig('fig/acp_instances_2nd_plan')
plt.close(fig)


print("###################################")
#Q5


# print correlations between factors and original variables
sqrt_eigval = np.sqrt(eigval)
corvar = np.zeros((p,p))
for k in range(p):
    corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]
# print(corvar)
# lines: variables
# columns: factors


correlation_circle(data,p,0,1)

correlation_circle(data,p,2,3)

print("###################################")
#Q6
from R_square_clustering import r_square

# ~ # choix du cluster effet coude (elbow)
lst_k = range(2, 20)
lst_rsq = []
for k in lst_k:
    est = KMeans(n_clusters=k)
    est.fit(standarScaledData)

    # est.labels = KMeans -> cluster labels for each instance of data
    #			  Purity -> predected labels
    lst_rsq.append(r_square(standarScaledData, est.cluster_centers_, est.labels_, k))  # 7 cluser pck coef trés grand 0.9

fig = plt.figure()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.xlabel('k')
plt.ylabel('RSQ')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('fig/k-means_elbow_method')
plt.close()



#Q7
listCountry = ['France', 'Mexico', 'Bulgaria']
kmeans = KMeans(8, random_state=0)
kmeans.fit_transform(standarScaledData)

for country in listCountry:
    clusterValues = data['Country Name'].values
    indexCountry = np.where(clusterValues == country)
    indexCountry = indexCountry[0][0]
    print("Country : ", country ," ||  Cluster number", kmeans.labels_[indexCountry])
    index = 0
    listOfCountriesCluster = []
    for clusterNumber in kmeans.labels_:
        if (clusterNumber == kmeans.labels_[indexCountry]):
            listOfCountriesCluster.append(data['Country Name'].values[index])
        index = index + 1

    print("\t Liste of countries in the same Cluster :",  listOfCountriesCluster)


#Q8
from scipy.cluster.hierarchy import dendrogram, linkage

lst_labels = list(map(lambda pair: pair[0]+str(pair[1]), zip(data['Country Name'].values,data.index)))
linkage_matrix = linkage(standarScaledData, 'ward')
fig = plt.figure()
dendrogram(
    linkage_matrix,
    color_threshold=0,
    labels=lst_labels
)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.savefig('fig/hierarchical-clustering')
plt.close()

#Q9

#coupure 5 clusters verifier avec les graphs avant

# ? pas d'achat
