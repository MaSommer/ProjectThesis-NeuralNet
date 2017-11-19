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
hidden_layer_dimension = [400,30]
time_lags_sp = 0
time_lags_ftse = 10
one_hot_vector_interval = [-0.000, 0.000]
#keep_probability_dropout = [1.0, 1.0] #first element is input layer and second is hidden layers

 #Data set specific
from_date =  "01.07.2008"
number_of_trading_days = 2000
attributes_input = ["op", "cp"]
selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
number_of_networks = 4
epochs = 40
number_of_stocks = 100


 #Training specific
learning_rate =0.1
minibatch_size = 10

rf_rate = 1.02

nr_of_runs = 10
global_run_nr = 1
soft_label = False
soft_label_percent = 1.0


selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, from_date, "S&P500_new.txt", 7,
                                     number_of_trading_days, normalize_method="minmax", start_time=time.time())

run_description = "Keep probability 1.0 and one hot vector 0.000 -> 0.006 with 0.003 in step. Soft label = [True, false]"

# parser = argparse.ArgumentParser()
# parser.add_argument('user_name')
# parser.add_argument('user_pwd')
# usr_pwd = parser.parse_args()
# print(usr_pwd)
# username = getattr(usr_pwd,'user_name')
# pwd = getattr(usr_pwd,'user_pwd')

soft_label_list = [True, False]
start_one_hot_interval = [-0.00, 0.00]


for soft_label in soft_label_list:
    start_keep_prob = [0.20, 0.20]
    for keep_prob_h1 in range(4):
        for keep_prob_h2 in range(8):
            for run_nr in range(1, nr_of_runs+1):
                time_start = time.time()
                test = run.Run(activation_functions, hidden_layer_dimension, time_lags_sp, time_lags_ftse, start_one_hot_interval, number_of_networks, start_keep_prob,
                               from_date, number_of_trading_days, attributes_input, number_of_stocks,
                               learning_rate, minibatch_size, epochs, rf_rate, global_run_nr, copy.deepcopy(sp500), soft_label, soft_label_percent, run_description)
                test.run_portfolio_in_parallell()
                time_end = time.time()
                print("--- Run " + str(global_run_nr) + " took %s seconds ---" % (time_end - time_start))
                global_run_nr += 1
            start_keep_prob[1] += 0.10
        start_keep_prob[0] += 0.20
        start_keep_prob[1] = 0.40


