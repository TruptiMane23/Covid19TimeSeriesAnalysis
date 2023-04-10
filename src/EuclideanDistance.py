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
from sklearn.metrics.pairwise import pairwise_distances



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
        plt.legend()
        plt.show()
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

    
    #-------------------------------Euclidean Distance--------------
    #print("column----------------")
    #print(seriesNames[0])
    #print(seriesNames[4])
    #arr1 = np.array(normalizedDataFrame[0].values)
    #arr1.shape = len(interpolatedDataframe[0])           # 1135
    #arr2 = np.array(normalizedDataFrame[4].values)
    #arr2.shape = len(interpolatedDataframe[0])
    #dist = distance.euclidean(arr1, arr2)
    #print("------Euclidean distance between Afganistan and India-----")
    #print(dist)

    #-------------------------------Euclidean Matrix-----------------
    print("-----Euclidean Distance Matrix after normalization- Bangladesh, Brazil, GER, IND -------")
    print(" ")
    allseriesDf = np.array(normalizedDataFrame)
    allseriesDf.shape = (len(normalizedDataFrame),len(interpolatedDataframe[0]))                       # (5,1135)
    mtx = squareform(pdist(allseriesDf))
    
    print(mtx)


    #-------------------------------DTW----------------------
    #dtwDistance = dtw.distance(arr1, arr2)
    #print("---------DTW distance between Afganistan and India--------")
    #print(dtwDistance)
    #path = dtw.warping_path(arr1, arr2)
    #dtwvis.plot_warping(arr1, arr2, path, filename="../img/dtw-AFG-IND.png")

    #-------------------------------DTW  Matrix-----------------
    print("------------DTW Distance Matrix - Bangladesh, Brazil, GER, IND--------")
    print(" ")
    ds = dtw.distance_matrix_fast(allseriesDf)


    print(ds)

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


    return normalizedDataFrame, seriesNames, interpolatedDataframe, count_Of_Series

if __name__ == '__main__':
    columns = [
        'date',
        'new_deaths', # 'daily_deaths'
    ]
    data, seriesNames, interpolatedDataframe, count_Of_Series = get_data(columns=columns)