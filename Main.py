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


activation_functions = ["tanh", "tanh", "tanh", "relu", "tanh", "sigmoid"]
hidden_layer_dimension = [300,50]
time_lags = 3
one_hot_vector_interval = [-0.07, 0.07]
keep_probability_dropout =0.80

 #Data set specific
from_date =  "01.01.2010"
number_of_trading_days = 1000
attributes_input = ["op", "cp"]
selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
number_of_networks = 2
epochs = 10
number_of_stocks =100


 #Training specific
learning_rate =0.1
minibatch_size = 10

rf_rate = 0.02

nr_of_runs = 3

for run_nr in range(1, nr_of_runs+1):
    test = run.Run(activation_functions, hidden_layer_dimension, time_lags, one_hot_vector_interval, number_of_networks, keep_probability_dropout,
               from_date, number_of_trading_days, attributes_input, number_of_stocks,
               learning_rate, minibatch_size, epochs, rf_rate, run_nr)
    test.run_portfolio_in_parallell()