import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import os
import Result as res
import NetworkManager as nm

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

    def run(self):
        f = open("res.txt", "w");
        selected = self.generate_selected_list()
        number_of_stocks_to_test = 4
        days = []
        day_returns = []

        for stock_nr in range(0, number_of_stocks_to_test):
            selected[stock_nr] = 1
            selectedFTSE100 = selected
            network_manager = nm.NetworkManager(self, selectedFTSE100, stock_nr)
            result = network_manager.build_networks(number_of_networks=2, epochs=40)
            result_string = result.genereate_result_string()

            self.write_result_to_file(result_string, stock_nr)
            selected[stock_nr] = 0

    def write_result_to_file(self, result_string, stock):
        f = open("res.txt", "a");
        f.write("REUSLT FOR " + str(stock) + "\n" + str(result_string) + "\n\n")  # python will convert \n to os.linesep
        f.close()

    def generate_selected_list(self):
        selected = []
        for i in range(0, 300):
            selected.append(0)
        return selected


main = Main()
main.run()