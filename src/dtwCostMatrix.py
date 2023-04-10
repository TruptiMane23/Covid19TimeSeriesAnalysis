from scipy.spatial import distance
import pandas as pd
import seaborn as sbn 
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import pdist, squareform
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")



def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = (x[j]-y[i])**2
    return dist


def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0,0] = distances[0,0]
    
    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
        
    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]  

    # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            ) + distances[i, j] 
            
    return cost


def _getCostMatrix() :
    #x = [1, 0, 1.5, 2, 1, 2, 0]
    #y = [0, 3, 1, 2, 1, 2]
   
    y = [0, 3, 1, 2, 1, 2]
    x = [1, 0, 1, 2, 1, 2, 0] 
    
    y = np.array([0, 3, 1, 2, 1, 2])
    x = np.array([1, 0, 1, 2, 1, 2, 0])
    


    
    arr1 = np.array(x)
    arr1.shape = (7, 1)   
    arr2 = np.array(y)
    arr2.shape = (6, 1)  

    dtw_distance, warp_path = fastdtw(arr1,arr2, dist=euclidean)

    cost_matrix = compute_accumulated_cost_matrix(x, y)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = sbn.heatmap(cost_matrix, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
    ax.invert_yaxis()

    # Get the warp path in x and y directions
    path_x = [p[0] for p in warp_path]
    path_y = [p[1] for p in warp_path]

    # Align the path from the center of each cell
    path_xx = [x+0.5 for x in path_x]
    path_yy = [y+0.5 for y in path_y]

    ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.2)

    fig.savefig("ex1_heatmap.png", **savefig_options)

    print("DTW distance: ", dtw_distance)
    print("Warp path: ", warp_path)


if __name__ == '__main__':

    _getCostMatrix()