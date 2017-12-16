import Run as run
import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import copy
import argparse


activation_functions = ["tanh", "tanh", "tanh", "tanh", "tanh", "tanh"]
hidden_layer_dimension = [300, 40]
time_lag_sp = 0
time_lags_ftse = 10
one_hot_vector_interval = [-0.000, 0.000]
keep_probability_dropout = [0.4, 0.6, 0.6] #first element is input layer and second is hidden layers

 #Data set specific
from_date =  "01.10.2008"
number_of_trading_days = 2000
attributes_input = ["op", "cp"]
selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
number_of_networks = 4
epochs = 40
number_of_stocks = 100


 #Training specific
learning_rate =0.01
minibatch_size = 10

rf_rate = 1.02

nr_of_runs = 3
global_run_nr = 1
soft_label = True
soft_label_percent = 1.0


selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, from_date, "S&P500_new.txt", 7,
                                     number_of_trading_days, normalize_method="minmax", start_time=time.time())

selectedFTSE = [0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	1,	1,	1,	1,	1,
                0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,
                1,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	1,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	1,	0,	1,	1,	0,	0,	1,	0,	1,	1,	0,	0]

run_description = "Different time lags"

# parser = argparse.ArgumentParser()
# parser.add_argument('user_name')
# parser.add_argument('user_pwd')
# usr_pwd = parser.parse_args()
# print(usr_pwd)
# username = getattr(usr_pwd,'user_name')
# pwd = getattr(usr_pwd,'user_pwd')

soft_label_list = [True, False]
start_one_hot_interval = [-0.00, 0.00]

h1_start = 300
h2_start = 0
h3_start = 0


epochs = [40]
from_dates = ["01.04.2008", "01.08.2008", "01.09.2008", "01.11.2008", "01.12.2008"]
time_lags_sp = [1, 2, 3, 4]

for time_lag_sp in time_lags_sp:
    for epoch in epochs:
        for run_nr in range(1, nr_of_runs+1):
            time_start = time.time()
            test = run.Run(activation_functions, hidden_layer_dimension, time_lag_sp, time_lags_ftse, start_one_hot_interval, number_of_networks, keep_probability_dropout,
                           from_date, number_of_trading_days, attributes_input, number_of_stocks,
                           learning_rate, minibatch_size, epoch, rf_rate, global_run_nr, copy.deepcopy(sp500), selectedFTSE, soft_label, soft_label_percent, run_description)
            test.run_portfolio_in_parallell()
            time_end = time.time()
            print("--- Run " + str(global_run_nr) + " took %s seconds ---" % (time_end - time_start))
            global_run_nr += 1





