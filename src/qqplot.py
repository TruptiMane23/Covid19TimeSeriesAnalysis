import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datasetkgModified import get_data


def _qqplot(data) :
    arr = np.array(data)
    arr.shape = (count_Of_Series, 1158) 
    fig = sm.qqplot(arr[0], line='45')
    plt.show()





if __name__ == '__main__':

    columns = [
        'date',
        'new_deaths',
    ]

    data, seriesNames, interpolatedDataframe, count_Of_Series = get_data(columns=columns)

    _qqplot(data)

    