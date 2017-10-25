import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import os
import numpy as np
import StockResult as res
import NetworkManager as nm
import matplotlib.pyplot as plt
import numpy as np



#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

#Standarized names for learningn method:        "gradient_decent" - Gradient decent

class Main():

    def __init__(self):
        self.start_time = time.time()
        self.fromDate = "01.01.2008"
        self.number_of_trading_days = 2000
        self.attributes_input = ["op", "cp"]
        self.attributes_output = ["ret"]
        self.one_hot_vector_interval = [-0.000, 0.000]

        self.time_lags = 3
        self.one_hot_size = 3

        self.keep_probability_for_dropout = 0.80

        self.portfolio_day_up_returns = []
        self.portfolio_day_down_returns = []

        self.learning_rate = 0.1
        self.minibatch_size = 10
        self.activation_functions = ["tanh", "tanh", "tanh", "sigmoid", "sigmoid", "relu", "relu", "relu", "relu", "relu"]
        self.initial_weight_range = [-1.0, 1.0]
        self.initial_bias_weight_range = [0.0, 0.0]
        self.cost_function = "cross_entropy"
        self.learning_method = "gradient_decent"
        self.validation_interval = None
        self.show_interval = None
        self.softmax = True

        self.hidden_layer_dimensions = [100,20]
        self.selectedSP500 = []
        self.selectedSP500 = ssr.readSelectedStocks("S&P500.txt")

        self.sp500 = pi.InputPortolfioInformation(self.selectedSP500, self.attributes_input, self.fromDate, "S&P500.txt", 7,
                                             self.number_of_trading_days, normalize_method="minmax", start_time=self.start_time)
        self.testing_days_list = []
        self.stock_results = []

    def run_portfolio(self):
        self.f = open("res.txt", "w");
        selectedFTSE100 = self.generate_selected_list()
        testing_size = 0
        number_of_stocks_to_test = 1
        #array with all the StockResult objects
        for stock_nr in range(0, number_of_stocks_to_test):
            selectedFTSE100[stock_nr] = 1
            network_manager = nm.NetworkManager(self, selectedFTSE100, stock_nr)
            stock_result = network_manager.build_networks(number_of_networks=4, epochs=40)
            result_string = stock_result.genereate_result_string()

            self.stock_results.append(stock_result)

            if (stock_nr == 0):
                self.day_list = network_manager.day_list
            print(result_string)
            self.write_result_to_file(result_string, stock_nr)
            selectedFTSE100[stock_nr] = 0

        self.write_long_results()
        self.write_short_results()
        self.print_portfolio_return_graph()
        self.f.close()

    def print_portfolio_return_graph(self):
        if (len(self.stock_results) > 0):
            portolfio_day_returns = self.find_portofolio_day_to_day_return(self.stock_results)
            portfolio_day_returns_as_percentage = self.make_return_percentage(portolfio_day_returns)
            standard_deviation_of_returns = np.std(self.convert_accumulated_portfolio_return_to_day_returns(portolfio_day_returns))
            self.write_portfolio_results(portfolio_day_returns_as_percentage[-1], standard_deviation_of_returns)

            day_list_without_jumps = self.make_day_list_without_day_jumps(len(self.day_list))
            plt.plot(day_list_without_jumps, portfolio_day_returns_as_percentage)
            self.scatter_plot_to_mark_test_networks(portfolio_day_returns_as_percentage)
            plt.show()
        else:
            raise ValueError("No stocks in result list")

    def scatter_plot_to_mark_test_networks(self, portolfio_day_returns):
        testing_sizes = self.stock_results[0].testing_sizes
        total_test = 0
        for test_size in testing_sizes:
            total_test += test_size
            ret = portolfio_day_returns[int(total_test)-1]
            plt.scatter(total_test-1, ret)

    def write_long_results(self):
        self.f = open("res.txt", "a")
        tot_up_return = self.get_portfolio_up_return()
        up_std = np.std(self.portfolio_day_up_returns)

        self.f.write("\nResults for long strategy:\n" )
        self.f.write("Return: " + str(tot_up_return)+"\n")
        self.f.write("Standard deviation: " + str(up_std) + "\n")
        self.f.close()

    def write_short_results(self):
        self.f = open("res.txt", "a")
        tot_down_return = self.get_portfolio_down_return()
        down_std = np.std(self.portfolio_day_down_returns)

        self.f.write("\nResults for short strategy:\n" )
        self.f.write("Return: " + str(tot_down_return)+"\n")
        self.f.write("Standard deviation: " + str(down_std) + "\n")
        self.f.close()

    def write_portfolio_results(self, over_all_portfolio_return, standard_deviation_of_returns):
        self.f = open("res.txt", "a")
        self.f.write("\n\nPORTFOLIO RETURN: " + "{0:.4f}%".format(over_all_portfolio_return)+
                     "\n" + "PORTFOLIO STANDARD DEVIATION: " + "{0:.4f}".format(standard_deviation_of_returns))
        self.f.close()

    def write_result_to_file(self, result_string, stock):
        self.f = open("res.txt", "a")
        self.f.write(result_string + "\n\n") # python will convert \n to os.linesep
        self.f.close()

    def generate_selected_list(self):
        selected = []
        for i in range(0, 300):
            selected.append(0)
        return selected

    def find_portofolio_day_to_day_return(self, stock_results):
        portfolio_day_returns = []
        day = 0
        for i in range(0, len(stock_results[0].day_returns_list)):
            total_return_for_day = 0
            for stock_result in stock_results:
                total_return_for_day += stock_result.day_returns_list[day]
            portfolio_day_returns.append(float(total_return_for_day)/float(len(stock_results)))
            day += 1
        return portfolio_day_returns

    def collect_portfolio_day_down_returns(self, stock_results): # Does not return anything
        for stock_result in stock_results:
            for ret in stock_result.get_day_down_returns():
                self.portfolio_day_down_returns.append(ret)

    def collect_portfolio_day_up_returns(self, stock_results): #Does not return anything
        for stock_result in stock_results:
            for ret in stock_result.get_day_up_returns():
                self.portfolio_day_up_returns.append(ret)

    def get_portfolio_up_return(self): #calculates total return on long strategy for portfolio
        tot_up_ret = 1.00
        if(len(self.portfolio_day_up_returns) == 0):
            self.collect_portfolio_day_up_returns(self.stock_results)

        for ret in self.portfolio_day_up_returns:
            tot_up_ret *= ret

        return tot_up_ret

    def get_portfolio_down_return(self): #calculates total return on short strategy for portfolio
        tot_down_ret = 1.00
        if(len(self.portfolio_day_down_returns) == 0):
            self.collect_portfolio_day_down_returns(self.stock_results)

        for ret in self.portfolio_day_down_returns:
            tot_down_ret *= ret

        return tot_down_ret


    def update_day_returns(self, day_returns):
        current_return = 1.0
        if (len(self.day_returns_list) > 0):
            current_return = self.day_returns_list[-1]
        for ret in range (0, len(day_returns)):
            current_return *= ret
            self.day_returns_list.append(current_return)

    def make_return_percentage(self, return_list):
        for i in range(0, len(return_list)):
            ret = (return_list[i]-1)*100
            return_list[i] = ret
        return return_list

    def make_day_list_without_day_jumps(self, length):
        new_day_list = []
        for i in range(0, length):
            new_day_list.append(i)
        return new_day_list


    def convert_accumulated_portfolio_return_to_day_returns(self, accumulated_returns):
        day_returns = []
        for i in range(1, len(accumulated_returns)):
            day_return = accumulated_returns[i]/accumulated_returns[i-1]
            day_returns.append(day_return)
        return day_returns

main = Main()
main.run_portfolio()