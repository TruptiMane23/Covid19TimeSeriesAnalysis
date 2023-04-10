from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as colorMap
import numpy as np
from datasetkgModified import get_data
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans 

def _silhouetePlot(count_Of_Series, data, nrows) :
    for k in [2,3,4,5,6,7]:
        arr = np.array(data)
        arr.shape = (count_Of_Series, nrows)
        plt.figure(figsize=(6, 5))
        plt.xlim([-1, 1])

        model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", random_state=1 )
        kMeanslabels = model.fit(data).predict(data)

        avgSilhouetteScore = silhouette_score(arr, kMeanslabels)
        print("The average silhouette score for ",k , "clusters is", avgSilhouetteScore)

        sampleSilhouetteScore = silhouette_samples(arr, kMeanslabels)

        yStart = 10
        for cluster in range(k):
            individualSilhouetteScore = sampleSilhouetteScore[kMeanslabels == cluster]
            individualSilhouetteScore.sort()
            print("-------After Sort---------")
            print(individualSilhouetteScore)

            clusterSize = len(individualSilhouetteScore)
            print("-------------Cluster Size----------------")
            print(clusterSize)
            yEnd = yStart + clusterSize

            clr = colorMap.nipy_spectral(float(cluster) / k)
            plt.fill_betweenx(np.arange(yStart, yEnd),0,individualSilhouetteScore,alpha=0.6,edgecolor=clr,facecolor=clr)

            plt.text(-0.05, (0.5 * clusterSize) + yStart, str(cluster))
            yStart = yEnd + 10

        plt.xlabel("-------------The silhouette coefficient------------")
        plt.ylabel("-------------Clusters---------------")
        plt.suptitle("The Silhouette plot for K clusters")
        plt.axvline(x=avgSilhouetteScore, linestyle="--", color="red")

    plt.show()

if __name__ == '__main__':

    columns = [
        'date',
        'new_deaths',
    ]

    data, seriesNames, interpolatedDataframe, count_Of_Series = get_data(columns=columns)

    _silhouetePlot(count_Of_Series, data, len(interpolatedDataframe[0]))