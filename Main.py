import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import os
import StockResult as res
import NetworkManager as nm
import matplotlib.pyplot as plt


#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

#Standarized names for learningn method:        "gradient_decent" - Gradient decent

class Main():

    def __init__(self):
        self.start_time = time.time()
        self.fromDate = "01.08.2005"
        self.number_of_trading_days = 120
        self.attributes_input = ["op", "cp"]
        self.attributes_output = ["ret"]
        self.one_hot_vector_interval = [-0.003, 0.003]

        self.time_lags = 3
        self.one_hot_size = 3

        self.learning_rate = 0.1
        self.minibatch_size = 10
        self.activation_functions = ["relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu"]
        self.initial_weight_range = [-1.0, 1.0]
        self.initial_bias_weight_range = [0.0, 0.0]
        self.cost_function = "cross_entropy"
        self.learning_method = "gradient_decent"
        self.validation_interval = None
        self.show_interval = None
        self.softmax = True

        self.number_of_networks_for_each_stock = 10
        self.hidden_layer_dimensions = [100, 60, 20]

        self.selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
        self.sp500 = pi.InputPortolfioInformation(self.selectedSP500, self.attributes_input, self.fromDate, "S&P500.txt", 7,
                                             self.number_of_trading_days, normalize_method="minmax", start_time=self.start_time)
        self.testing_days_list = []
        self.stock_results = []

    def run_portfolio(self):
        self.f = open("res.txt", "w");
        selectedFTSE100 = self.generate_selected_list()
        number_of_stocks_to_test = 2
        #array with all the StockResult objects
        for stock_nr in range(0, number_of_stocks_to_test):
            selectedFTSE100[stock_nr] = 1
            network_manager = nm.NetworkManager(self, selectedFTSE100, stock_nr)
            stock_result = network_manager.build_networks(number_of_networks=2, epochs=40)
            result_string = stock_result.genereate_result_string()
            self.stock_results.append(stock_result)

            if (stock_nr == 0):
                self.day_list = network_manager.day_list
            self.write_result_to_file(result_string, stock_nr)
            selectedFTSE100[stock_nr] = 0

        self.print_portfolio_return_graph()
        self.f.close()

    def print_portfolio_return_graph(self):
        if (len(self.stock_results) > 0):
            portolfio_day_returns = self.find_portofolio_day_to_day_return(self.stock_results)
            self.write_portfolio_return(portolfio_day_returns[-1])
            day_list_without_jumps = self.make_day_list_without_day_jumps(len(self.day_list))
            plt.plot(day_list_without_jumps, portolfio_day_returns)

            plt.show()
        else:
            raise ValueError("No stocks in result list")


    def write_portfolio_return(self, over_all_portfolio_return):
        self.f.write("\n\nOVER ALL PORTFOLIO RETURN: " + str(over_all_portfolio_return))

    def write_result_to_file(self, result_string, stock):
        self.f = open("res.txt", "a");
        self.f.write("REUSLT FOR " + str(stock) + "\n" + str(result_string) + "\n\n")  # python will convert \n to os.linesep

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
        return self.make_return_percentage(portfolio_day_returns)

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



main = Main()
main.run_portfolio()