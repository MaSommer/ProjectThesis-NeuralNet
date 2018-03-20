import numpy as np
from sklearn.decomposition import PCA
import statsmodels.tsa.stattools as ts
import time
import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi


 #Data set specific
from_date =  "01.10.2008"
number_of_trading_days = 1000
attributes_input = ["op", "cp"]


selectedSP500 = ssr.readSelectedStocks("S&P500v2.txt")
sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, from_date, "S&P500_new.txt", 7,
                                     number_of_trading_days, normalize_method="minmax", start_time=time.time())

data_set = {}
stock_nr = 0

input_data = sp500.portfolio_data

for key in input_data:
    data_set["DataType-" + str(key)] = {}
    for stock_nr in range(0, len(input_data[key][0])):
        data_set["DataType-" + str(key)]["Stock-" + str(stock_nr)] = []
    for data in input_data[key]:
        for stock_nr in range(0, len(data)):
            data_set["DataType-" + str(key)]["Stock-" + str(stock_nr)].append(data[stock_nr])


def pca(data_set):
    X = []
    for key in data_set:
        for stock_nr in data_set[key]:
            data_list = data_set[key][stock_nr]
            X.append(data_list)
    pca = PCA(n_components=6)
    pca.fit(X)
    print("Explained variance:")
    print(pca.explained_variance_ratio_)
    print("Eigenvalues")
    print(pca.singular_values_)


def dicky_fuller_test(data_set):
    dicky_fuller_results = {}

    for key in data_set:
        dicky_fuller_results[key] = {}
        for stock_nr in data_set[key]:

            dicky_fuller_results[key][stock_nr] = ts.adfuller(data_set[key][stock_nr], 5)


    for key in dicky_fuller_results:
        for stock_nr in dicky_fuller_results[key]:
            statistic = dicky_fuller_results[key][stock_nr][0]
            treshold = dicky_fuller_results[key][stock_nr][4]["1%"]
            #print("ADF: " + str(dicky_fuller_results[key][stock_nr]))
            if (statistic >= treshold):
                print("ADF: " + str(dicky_fuller_results[key][stock_nr]))


pca(data_set)

