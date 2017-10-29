import Run as run
import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import copy
import os
import numpy as np
import StockResult as res
import NetworkManager as nm
import matplotlib.pyplot as plt
import numpy as np
import ExcelFormatter as excel
import HyperParamResult as hpr


activation_functions = ["tanh", "tanh", "tanh", "tanh", "tanh", "tanh"]
hidden_layer_dimension = [500,40]
time_lags = 1
one_hot_vector_interval = [-0.005, 0.005]
keep_probability_dropout =0.80

 #Data set specific
from_date =  "01.01.2012"
number_of_trading_days = 700
attributes_input = ["op", "cp"]
selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
number_of_networks = 1
epochs = 40
number_of_stocks =100


 #Training specific
learning_rate =0.1
minibatch_size = 10

rf_rate = 1.02

nr_of_runs = 10
global_run_nr = 1


selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, from_date, "S&P500.txt", 7,
                                     number_of_trading_days, normalize_method="minmax", start_time=time.time())

for time_lag in range(0, time_lags):
    for run_nr in range(1, nr_of_runs+1):
        time_start = time.time()
        test = run.Run(activation_functions, hidden_layer_dimension, time_lag, one_hot_vector_interval, number_of_networks, keep_probability_dropout,
                   from_date, number_of_trading_days, attributes_input, number_of_stocks,
                   learning_rate, minibatch_size, epochs, rf_rate, global_run_nr, copy.deepcopy(sp500))
        test.run_portfolio_in_parallell()
        time_end = time.time()
        print("--- Run " + str(global_run_nr) + " took %s seconds ---" % (time_end - time_start))
        global_run_nr += 1