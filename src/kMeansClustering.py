import math
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans         # This uses kmeans algorith with DTW - dynamic time warping
from datasetkgModified import get_data
#from tslearn.barycenters import dtw_barycenter_averaging
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
from pandas.plotting import lag_plot
import time


# K-Means clustering ****************************************************************************************************

# To determine initial no of clusters
def determine_no_of_clusters_elbow(data) :
    start_time = time.time()
    first_time = start_time
    timeArray = []
    withinClusterSumOfSqr = []
    noOfIterations = []
    countOfCluster = range(1,10)
    for k in countOfCluster:
        
        kmeanElbowModel = TimeSeriesKMeans(n_clusters=k, metric = 'euclidean', random_state=1) #metric = 'dtw'
        kmeanElbowModel.fit_predict(data)
        withinClusterSumOfSqr.append(kmeanElbowModel.inertia_)
        print("printing iterations ----------------")
        print(kmeanElbowModel.n_iter_)
        noOfIterations.append(kmeanElbowModel.n_iter_)
        end_time = time.time()
        timeArray.append(end_time - start_time)
        start_time = time.time()

    plt.figure(figsize=(10,7))
    plt.title('The optimal number of clusters using Elbow method')
    plt.plot(countOfCluster, withinClusterSumOfSqr, 'bx-')
    #plt.plot(countOfCluster, withinClusterSumOfSqr, '-b', label = "within cluster sum of square")
    #plt.plot(countOfCluster, noOfIterations, '-r', label = "no of iterations" )
    #plt.plot(countOfCluster, timeArray, '-r', label = "Time taken" )
    plt.xlabel('Clusters')
    plt.ylabel('Within cluster sum of square')
    #plt.ylim(0,10)
    #plt.legend()
    #print("--- %s seconds ---" % (time.time() - first_time))
    plt.show()


# To get cluster labels for KMeans - i.e determine which series belong to which cluster 
def _getLabels(data, nrows):
    model = TimeSeriesKMeans(n_clusters=6, metric="euclidean", random_state=1) #max_iter=10, euclidean
    lables_array_of_shape = model.fit(data).predict(data)
    print("----------KMeans lables----------")
    print(lables_array_of_shape)
    print("----------KMeans cluster centers----------")
    print(model.cluster_centers_)
    clusterCenters = model.cluster_centers_

    #print("-------------Silhouette Score for 6 clusters-----------------")
    #arr = np.array(data)
    #arr.shape = (count_Of_Series, nrows)          
    #score = silhouette_score(arr, lables_array_of_shape, metric='euclidean')
    #print(score)

    return lables_array_of_shape, clusterCenters




# Agglomerative clustering ***********************************************************************************************
 
def plotDendogram(data, count_Of_Series, nrows) :  
    arr = np.array(data)
    arr.shape = (count_Of_Series, nrows)                                                                            #  Old(23*138)    newDataset(23*1158)
    print(arr.shape)
    print(arr.ndim)
    linked = linkage(arr, method ='single', metric = 'cityblock', optimal_ordering=False)   # method - single, cmplete, average, ward   metric = euclidean, cityblock

    #Plot Dendrogram
    plt.figure(figsize=(5, 3))
    plt.title("ward linkage - euclidean distance")
    dendogramDictionry = dendrogram(linked, orientation='top', 
                                    distance_sort='descending', show_leaf_counts=True)
    plt.ylabel("Euclidean distance")
    plt.xlabel("ward linkage clusters")
    plt.show()

    #print("----------printing unique color list----------")
    #unique_colors = set(dendogramDictionry['color_list'])
    #print(unique_colors)
    #print("----------printing optimal no of clusters-------")
    #optimal_no_of_clusters = len(unique_colors)-1
    #print(optimal_no_of_clusters)
    

    # Cluster labels for agglomerative clustering
    #agglomerative_cluster = AgglomerativeClustering(distance_threshold=8, n_clusters=None, metric='euclidean', linkage='complete')                                                                   
    agglomerative_cluster = AgglomerativeClustering(n_clusters=3, compute_distances=True, metric='manhattan', linkage='single')
    labels = agglomerative_cluster.fit_predict(arr)
    
    
    print("-----------Agglomerative cluster labels--------------")
    print(labels)


    distMatx = agglomerative_cluster.distances_
    print("-------------printing dist matxxxxxxxxxx----------------")
    print(distMatx)

    print("-------------Silhouette Score for ward linkage-----------------")
    arr = np.array(data)
    arr.shape = (count_Of_Series, nrows)          
    score = silhouette_score(arr, labels, metric='euclidean')
    print(score)

    #plt.scatter(data[8], data[11], c=['red','green'])
    #plt.show()
    #sns.scatterplot(arr,hue=labels)

    #plt.show()
    return labels

def _TestForPerformance(data, count_Of_Series, nrows) :
    arr = np.array(data)
    arr.shape = (count_Of_Series, nrows) 
    withinClusterSumOfSqr = []
    #noOfIterations = []
    #countOfCluster = range(1,10)
    #for k in countOfCluster:
    aggloModel = AgglomerativeClustering(distance_threshold=8, n_clusters=None, metric = 'euclidean', linkage='ward')
    aggloModel.fit_predict(arr)
    withinClusterSumOfSqr.append(aggloModel.distances_)
    print("printing iterations ----------------")
    #print(aggloModel.n_iter_)
    #noOfIterations.append(aggloModel.n_iter_)

    plt.figure(figsize=(10,7))
    plt.title('Test for Agglomerative clustering - Euclidean distance')
    plt.plot(aggloModel.n_clusters_, withinClusterSumOfSqr, '-b', label = "within cluster sum of square")
    #plt.plot(aggloModel.n_clusters_, noOfIterations, '-r', label = "no of iterations" )
    plt.xlabel('No of Clusters')
    plt.ylabel('Within Cluster Sum Of Square and No of iteration')
    plt.ylim(0,30)
    plt.legend()
    plt.show()



# cluster assignment based on cluster lables for both KMeans and Hierarchical 
def _assignSeriesToRespectivecluster(data, lables_array_of_shape) :
    clusterGrid =[]
    for j in range(7):
        cluster = ("cluster"+str(j)) 
        cluster = []
        clusterGrid.append(cluster)
    print("---------------All clusters Grid size------------")
    print(len(clusterGrid))

    i=0
    for series in data:
        for k in range(7):
            if(lables_array_of_shape[i] == k):
                clusterGrid[k].append(series)      
        i+=1

    print("----------Length of all clusters in grid-----------")
    for l in range(7):
        print(len(clusterGrid[l]))

    return clusterGrid


def _createNameGrid(lables, seriesNames):
    nameGrid = []
    labelArray0 = []
    labelArray1 = []
    labelArray2 = []
    labelArray3 = []
    labelArray4 = []
    labelArray5 = []
    labelArray6 = []
    nameGrid.append(labelArray0)
    nameGrid.append(labelArray1)
    nameGrid.append(labelArray2)
    nameGrid.append(labelArray3)
    nameGrid.append(labelArray4)
    nameGrid.append(labelArray5)
    nameGrid.append(labelArray6)

    for label in range(len(lables)):
        if(lables[label] == 0):
            labelArray0.append(seriesNames[label]) 
        if(lables[label] == 1):
            labelArray1.append(seriesNames[label]) 
        if(lables[label] == 2):
            labelArray2.append(seriesNames[label]) 
        if(lables[label] == 3):
            labelArray3.append(seriesNames[label]) 
        if(lables[label] == 4):
            labelArray4.append(seriesNames[label]) 
        if(lables[label] == 5):
            labelArray5.append(seriesNames[label]) 
        if(lables[label] == 6):
            labelArray6.append(seriesNames[label]) 
    label+=1
    print(nameGrid)

    return nameGrid



def _plot_clusters(clusterGrid, clusterGridSize, title, nameGrid):
    nrows=4
    ncols=2
    i=0
    figure, axs = plt.subplots(nrows, ncols, figsize=(10, 5))
    for row in range(4):
        for col in range(2):
            if i<clusterGridSize:          
                for series in clusterGrid[i]:
                    axs[row][col].plot(series.values) 
                    axs[row][col].set_title(nameGrid[i])  #label=nameGrid[i]          # "cluster - " + str(i)
                i+=1
    
    figure.suptitle(title)
    figure.tight_layout()
    plt.subplots_adjust(top=0.9,bottom=0.1)
    plt.legend()
    #plt.plot(model.clustercenters)
    plt.show()


def _plot_cluster_assignment(clusterGrid):
    arr1 = []
    arr2 = []
    for i in range(len(clusterGrid)):
        arr1.append("cluster"+str(i))

    for cluster in clusterGrid:
        arr2.append(len(cluster))
    
    #plt.bar(arr1,arr2)
    #plt.show()




if __name__ == '__main__':

    columns = [
        'date',
        'new_deaths', #'daily_deaths'
    ]

    data, seriesNames, interpolatedDataframe, count_Of_Series = get_data(columns=columns)

    

    # K means algorithm ----------------------------------------------------------------------
    determine_no_of_clusters_elbow(data)
#
 #  lables_array_of_shape, clusterCenters = _getLabels(data, len(interpolatedDataframe[0]))
#
    #nameGrid = _createNameGrid(lables_array_of_shape, seriesNames) 
##
    #clusterGrid = _assignSeriesToRespectivecluster(data, lables_array_of_shape)
##
    #_plot_clusters(clusterGrid, len(clusterGrid), "K means Clusters", nameGrid)
##
    #_plot_cluster_assignment(clusterGrid)

    #_scatterPlot(lables_array_of_shape, clusterCenters, interpolatedDataframe)
    



    # Heirarchical algorithm -------------------------------------------------------------------
    
    # 1. Agglomerative clustering using Eucliedean/Manhattan as distance metric and  single, complete, average, ward as linkage
    #labels  = plotDendogram(data, count_Of_Series, len(interpolatedDataframe[0]))
    #nameGrid = _createNameGrid(labels, seriesNames) 
#
    #clusterGrid = _assignSeriesToRespectivecluster(data, labels)
    #_plot_clusters(clusterGrid, len(clusterGrid), "Agglomerative Clustering", nameGrid)
    
    #_TestForPerformance(data, count_Of_Series, len(interpolatedDataframe[0]))
    
    
    
    #python.exe .\kMeansClustering.py
