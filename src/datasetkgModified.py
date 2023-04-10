import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler                  # for Normalization
#from minisom import MiniSom
#import xlsxwriter as excel
from scipy.spatial import distance

AllSeries = []
seriesNames = []
def _read_data(path, columns):
    #plt.figure(figsize=(15, 15))
    #plt.ylim(0, 4500)
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
    #print(count_of_series)
    return count_of_series


def _print_series_start_EndDate(AllSeries):
    position = 0
    for eachSeries in AllSeries:
        #print(str(position) + " --> " + str(eachSeries.index[0])+" -- "+str(eachSeries.index[-1]))
        position+=1


def _plot_data(AllSeries, count_Of_Series, title):
    nrows=6
    ncols=4
    i=0
    figure, axs = plt.subplots(nrows, ncols, figsize=(12, 7))
    for row in range(nrows): 
        for col in range(ncols):
            if i<count_Of_Series:
                axs[row][col].set_title(seriesNames[i])
                axs[row][col].plot(AllSeries[i].values)
                #print("Series name .......")
                #print(seriesNames[i])
                #print(AllSeries[i].values)
                i+=1
    figure.suptitle(title)
    figure.tight_layout()
    plt.subplots_adjust(top=0.9,bottom=0.1)
    #plt.show()


#To find minimum and maximum of date range of all series
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
    #print(len(interpolatedDataframe[0]))

    return interpolatedDataframe



# To normalize all series
normalizedDataFrame = [] 
def _normalize_data(interpolatedDataframe, count_Of_Series):
    for series in interpolatedDataframe:
        newSeries = pd.DataFrame(MinMaxScaler().fit_transform(series), 
                                 columns= series.columns, index=series.index)
        normalizedDataFrame.append(newSeries)

    print("-----------------After Normalization---------------")
    #print(normalizedDataFrame)

    #normalizedArr = np.array(normalizedDataFrame)
    #normalizedArr.shape = (count_Of_Series, len(normalizedDataFrame[0]))
    #df = pd.DataFrame(normalizedArr).T
    #print(df.shape)
    #df.to_csv("../dataset/Normalized.csv")
    

    return normalizedDataFrame


def get_data(
        #path = '../dataset/AllstatsData/',
        path = '../dataset/23OwidData/',
        columns = None):
    
    # To read data
    AllSeries, seriesNames = _read_data(path=path, columns=columns)


    # To count total number of Series 
    count_Of_Series = _countSeries(AllSeries)


    # To visualize each series start and end date 
    _print_series_start_EndDate(AllSeries)

    
    # To Plot all data
    title ="Original Dataset"
    _plot_data(AllSeries, count_Of_Series, title)


    #To find minimum and maximum of date range of all series
    start_date, end_date = _find_min_max_of_daterange(AllSeries, count_Of_Series)


    #To reindex all the series 
    reindexedDataframe = _reindex_AllSeries(start_date, end_date, AllSeries)


    #To interpolate missing values
    interpolatedDataframe = _interpolate_missing_values(reindexedDataframe)


    #To plot interpolation data 
    title ="Interpolated Data" 
    _plot_data(interpolatedDataframe, count_Of_Series, title)
    

    # To normalize data
    normalizedDataFrame = _normalize_data(interpolatedDataframe, count_Of_Series)

    # To plot Normalized data
    title ="Normalized Data" 
    _plot_data(normalizedDataFrame, count_Of_Series, title)


    return normalizedDataFrame, seriesNames, interpolatedDataframe, count_Of_Series



if __name__ == '__main__':
    columns = [
        'date',
        'new_deaths', # 'daily_deaths'
    ]
    data, seriesNames, interpolatedDataframe, count_Of_Series = get_data(columns=columns)



















#pip3.10.exe install pandas
#pip3.10.exe install matplotlib
#pip3.10.exe install scikit-learn - for sklearn.preprocessing
#pip3.10.exe install tslearn  - for all
#pip3.10.exe install dtaidistance 
#pip3.10.exe install statsmodels 

 
#python.exe .\kMeansClustering.py

# pip new version installed - 23.0.1      