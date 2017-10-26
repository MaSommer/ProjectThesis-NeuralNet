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


activation_functions = ["relu", "relu", "relu", "relu", "tanh", "sigmoid"]
hidden_layer_dimension = [100]
time_lags = 1
one_hot_vector_interval = [-0.000, 0.000]
keep_probability_dropout =0.90

 #Data set specific
from_date =  "01.01.2008"
number_of_trading_days = 100
attributes_input = ["op", "cp"]
selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
number_of_networks = 3
epochs = 10
number_of_stocks = 2


 #Training specific
learning_rate =0.1
minibatch_size = 10

nr_of_runs =2 #initially 1

for run_nr in range(1, nr_of_runs+1):
    print("\n*****--------RUN NR " + str(run_nr) + "--------*****\n")
    test = run.Run(activation_functions, hidden_layer_dimension, time_lags, one_hot_vector_interval, number_of_networks, keep_probability_dropout,
               from_date, number_of_trading_days, attributes_input, number_of_stocks,
               learning_rate, minibatch_size, epochs, run_nr)
    test.run_portfolio_in_parallell()