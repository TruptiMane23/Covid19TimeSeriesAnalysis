from scipy.spatial import distance
import pandas as pd
import seaborn as sbn 
import matplotlib.pyplot as plt
from datasetkgModified import get_data
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



def _getCostMatrix(normalizedDataFrame) :

    arr1 = np.array(normalizedDataFrame[0])
    arr1.shape = (1132, 1)   

    arr2 = np.array(normalizedDataFrame[1])
    arr2.shape = (1132, 1)  


    dtw_distance, warp_path = fastdtw(arr1,arr2, dist=euclidean)


    cost_matrix = compute_accumulated_cost_matrix(arr1, arr2)
#
#
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

    fig.savefig("ex2_heatmap.png", **savefig_options)

    print("DTW distance: ", dtw_distance)
    print("Warp path: ", warp_path)







AllSeries = []
seriesNames = []
def _read_data(path, columns):
    for file in os.listdir(path):
        df = pd.read_csv(path+file)
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        df = df.loc[:,[columns[0],columns[1]]]  # date, daily_deaths (new_deaths)
        df = df.set_index("date")
        df = df.sort_index()
        AllSeries.append(df)
        seriesNames.append(file.split('.')[0])

        df['new_deaths'].plot(label=file.split('.')[0])
        #plt.legend()
        #plt.show()
    return AllSeries, seriesNames

def _countSeries(AllSeries):
    print("------------Total number of series------------")
    count_of_series = len(AllSeries)
    print(count_of_series)
    return count_of_series

min_date =[]
max_date =[]
def _find_min_max_of_daterange(AllSeries, count_Of_Series):
    for series in AllSeries:
        min_date.append(series.index[0])
        max_date.append(series.index[-1])

    start_date = min(min_date)
    print("-------------Start date-------------")
    #print(start_date)
    
    end_date = max(max_date)
    print("-------------End date---------------")
    #print(end_date)
    return start_date, end_date



# Reindexes with NAN values
reindexedDataframe = []    
def _reindex_AllSeries(start_date, end_date, AllSeries):
    newIndexRange = pd.date_range(start_date, end_date)
    newCommonIndex = pd.Index(newIndexRange)
    for series in AllSeries:
        series = series.reindex(newCommonIndex)
        reindexedDataframe.append(series)
        print("---------------After Reindexing----------------")
        #print(series)
    return reindexedDataframe



# To interpolate - i.e to fill missing values with numbers
interpolatedDataframe = []
def _interpolate_missing_values(reindexedDataframe):
    for series in reindexedDataframe:
        series.interpolate(method='linear', limit_direction="both",inplace=True)
        interpolatedDataframe.append(series)
        print("--------------After Interpolation--------------")
        #print(series)    
    return interpolatedDataframe



# To normalize all series
normalizedDataFrame = [] 
def _normalize_data(interpolatedDataframe, AllSeries):
    for series in interpolatedDataframe:
        newSeries = pd.DataFrame(MinMaxScaler().fit_transform(series), 
                                 columns= series.columns, index=series.index)
        normalizedDataFrame.append(newSeries)

    print("-----------------After Normalization---------------")
    #print(normalizedDataFrame)

    return normalizedDataFrame


def get_data(
        path = '../dataset/EuclideanDistanceGer,Ind,Bra/',
        columns = None):
    
    # To read data
    AllSeries, seriesNames = _read_data(path=path, columns=columns)


    # To count total number of Series 
    count_Of_Series = _countSeries(AllSeries)



    #To find minimum and maximum of date range of all series
    start_date, end_date = _find_min_max_of_daterange(AllSeries, count_Of_Series)


    #To reindex all the series 
    reindexedDataframe = _reindex_AllSeries(start_date, end_date, AllSeries)


    #To interpolate missing values
    interpolatedDataframe = _interpolate_missing_values(reindexedDataframe)
    

    # To normalize data
    normalizedDataFrame = _normalize_data(interpolatedDataframe,AllSeries)

    _getCostMatrix(interpolatedDataframe)   

    return normalizedDataFrame, seriesNames, interpolatedDataframe, count_Of_Series, AllSeries

if __name__ == '__main__':
    columns = [
        'date',
        'new_deaths', # 'daily_deaths'
    ]
    normalizedDataFrame, seriesNames, interpolatedDataframe, count_Of_Series, AllSeries = get_data(columns=columns)

  
