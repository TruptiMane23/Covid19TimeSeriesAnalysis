import numpy as np
import matplotlib.pyplot as plt
from EuclideanDistance import get_data

from tslearn.barycenters import \
    euclidean_barycenter, \
    dtw_barycenter_averaging, \
    dtw_barycenter_averaging_subgradient, \
    softdtw_barycenter
from tslearn.datasets import CachedDatasets


def plot_helper(barycenter, data):
    for series in data:
            plt.plot(series.values.ravel(), "b-", alpha=.2)
    plt.plot(barycenter.ravel(), "r-", linewidth=1)


# plot the four variants with the same number of iterations and a tolerance of 1e-3 where applicable
def _plotBarycenter(data, nrows) :
    arr = np.array(data)
    arr.shape = (count_Of_Series, nrows)

    #ax1 = plt.subplot(1, 1, 1)
    #plt.title("Euclidean barycenter")
    #baryArry = euclidean_barycenter(arr)
    #plot_helper(baryArry, data)

    plt.subplot(1, 1, 1)
    plt.title("DTW Barycenter Averaging (DBA)")
    plot_helper(dtw_barycenter_averaging(arr, max_iter=50, tol=1e-3), data)
#
    #plt.subplot(4, 1, 1)
    #plt.title("DBA (subgradient descent approach)")
    #plot_helper(dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3))
#
    #plt.subplot(1, 1, 1)
    #plt.title("Soft-DTW barycenter")
    #plot_helper(softdtw_barycenter(arr, gamma=1., max_iter=50, tol=1e-3), data)

    # clip the axes for better readability
    #ax1.set_xlim([0, length_of_sequence])

    # show the plot(s)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    columns = [
        'date',
        'new_deaths',
    ]

    data, seriesNames, interpolatedDataframe, count_Of_Series = get_data(columns=columns)
    _plotBarycenter(data, len(interpolatedDataframe[0]))

    