from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler  
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import pdist, squareform

T1 = np.array([1,2,1,3,1,2,4,1,3,1,3,1,4,1,3,1,2,1,3,1,5,2,4,1,3,1,4,1,3,1,4]) # original
T2 = np.array([4,5,4,6,4,5,7,4,6,4,6,4,7,4,6,4,5,4,6,4,8,5,7,4,6,5,8,5,7,5,8]) # Value offset 
T3 = np.array([4,4,4,6,5,7,5,6,8,5,7,5,7,5,8,5,7,5,6,5,7,5,9,6,8,5,7,5,8,5,7]) # Time offset

#----Euclidean Distance for time and value offset time series--------
EuclTimeOffsetDist = distance.euclidean(T1, T3)
EuclValueOffsetDist = distance.euclidean(T1, T2)
print("---Euclidean distance for value offset Time series T1 and T2---")
#print(EuclTimeOffsetDist)
print(EuclValueOffsetDist)

#-----DTW Distance for time and value offset time series--------
dtwTimeOffDistance = dtw.distance(T1, T3)
dtwValueOffDistance = dtw.distance(T1, T2)
print("---DTW distance for value offset Time series T1 and T2---")
#print(dtwTimeOffDistance)
print(dtwValueOffDistance)
path = dtw.warping_path(T1, T3)
dtwvis.plot_warping(T1, T3, path, filename="../img/dtwTimeOffset-T1,T3.png")
