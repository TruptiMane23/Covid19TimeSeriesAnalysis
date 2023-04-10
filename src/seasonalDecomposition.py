import pandas as pd
import numpy as np
from matplotlib import pyplot
from datasetkgModified import get_data
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

def _seasonal_decomposition(interpolatedDataframe, normalizedData) :

    # Time series ETS decomposition ------------------------------------
    df = pd.read_csv('../dataset/DecompositionDataset.csv')
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, format='%d-%m-%Y')
    df = df.loc[:,['date', 'AFG-new_deaths', 'AFG-stringency_index', 'AFG-new_cases', 
                   'AFG-new_people_vaccinated_smoothed', 'VIE-new_deaths', 'VIE-stringency_index',
                    'VIE-new_cases', 'VIE-new_people_vaccinated_smoothed', 'BRA-new_deaths', 
                    'BRA-stringency_index', 'BRA-new_cases', 'BRA-new_people_vaccinated_smoothed', 
                    'IND-new_deaths', 'IND-stringency_index', 'IND-new_cases', 'IND-new_people_vaccinated_smoothed']] 

    
    df = df.set_index("date")
    df.sort_index(inplace=True)

    #df.plot()
    result = seasonal_decompose(df['AFG-new_deaths'], model='additive')
    result.plot()
    pyplot.show()




    # For normalization of raw data 
    newSeries = pd.DataFrame(MinMaxScaler().fit_transform(df), index=df.index)
    print("-----------------After Normalization---------------")
    print(newSeries)
    pyplot.scatter(newSeries[8], newSeries[11])
    pyplot.show()
#





    # impo ********************** imp **********************
    corr1 = df['BRA-new_deaths'].corr(df['IND-new_deaths'], method='pearson')               
    print("Correlation between Brazil new death cases and India new death cases---->")
    print(corr1)

    corr2 = df['BRA-new_deaths'].corr(df['BRA-new_people_vaccinated_smoothed'], method='pearson')
    print("Correlation between Brazil new death cases and new people vaccinated---->")
    print(corr2)

    corr3 = df['BRA-new_deaths'].corr(df['BRA-stringency_index'], method='pearson')
    print("Correlation between Brazil new death cases and stringency_index---->")
    print(corr3)

    corr4 = df['BRA-new_deaths'].corr(df['IND-new_deaths'], method='spearman')
    print("Correlation between Brazil new death cases and India new death cases---->")
    print(corr4)

    corr5 = df['BRA-new_deaths'].corr(df['BRA-new_people_vaccinated_smoothed'], method='spearman')
    print("Correlation between Brazil new death cases and new people vaccinated---->")
    print(corr5)

    corr6 = df['BRA-new_deaths'].corr(df['BRA-stringency_index'], method='spearman')
    print("Correlation between Brazil new death cases and stringency_index---->")
    print(corr6)


    #corr1 = df['new_deaths'].corr(df['stringency_index'], method='spearman')
    #print("spearman Correl between death and stringency is -")
    #print(corr1)
#
    #corr2 = df['new_deaths'].corr(df['stringency_index'], method='pearson')
    #print("pearson Correl between death and stringency is -")
    #print(corr2)
#
    #corr3 = df['new_deaths'].corr(df['stringency_index'], method='kendall')
    #print("kendall Correl between death and stringency is -")
    #print(corr3)


    #corr4 = df['new_cases'].corr(df['stringency_index'], method='spearman')
    #print("spearman Correl between new_cases and stringency is ..")
    #print(corr4)
#
    #corr5 = df['new_cases'].corr(df['stringency_index'], method='pearson')
    #print("pearson Correl between new_cases and stringency is ..")
    #print(corr5)
#
    #corr6 = df['new_cases'].corr(df['stringency_index'], method='kendall')
    #print("kendall Correl between new_cases and stringency is ..")
    #print(corr6)






    #df[['new_cases', 'stringency_index']].plot()
    #pyplot.show()









    #result = seasonal_decompose(interpolatedDataframe[15], model='additive')
    #result.plot()
    #pyplot.title("India ETS Decomposition")
    #pyplot.show()




    #arr = np.array(interpolatedDataframe)
    #arr.shape = (count_Of_Series, 1158)   
    #dat = pd.DataFrame(arr)
    #print("Correl is ..")
    #print(dat.corr())


    #df = pd.DataFrame(interpolatedDataframe[13])
    #corr = df['date'].corr(df['new_deaths'])
    #corr = (np.array(interpolatedDataframe[13].values)).corr(np.array(interpolatedDataframe[14].values))
    #print("Correl is ..")
    #print(corr)





if __name__ == '__main__':
    columns = [
        'date',
        'new_deaths', # 'daily_deaths'
    ]
    normalizedData, seriesNames, interpolatedDataframe, count_Of_Series = get_data(columns=columns)
    _seasonal_decomposition(interpolatedDataframe, normalizedData)
    #_seasonal_decomposition()

