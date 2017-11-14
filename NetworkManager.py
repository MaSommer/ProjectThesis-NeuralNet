import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import os
import StockResult as res
import sys



class NetworkManager():

    def __init__(self, run, selectedFTSE100, stock_nr, run_nr, soft_label = False, soft_label_percent = 1.0):
        self.selectedFTSE100 = selectedFTSE100
        self.sp500 = run.sp500

        self.stock_nr = stock_nr
        self.fromDate = run.fromDate
        self.number_of_trading_days = run.number_of_trading_days
        self.start_time = time.time()
        self.global_start_time = run.start_time
        self.time_lags_sp = run.time_lags_sp
        self.time_lags_ftse = run.time_lags_ftse
        self.one_hot_size = run.one_hot_size
        self.one_hot_vector_interval = run.one_hot_vector_interval
        self.minibatch_size = run.minibatch_size
        self.learning_rate = run.learning_rate
        self.activation_functions = run.activation_functions
        self.cost_function = run.cost_function
        self.learning_method = run.learning_method
        self.validation_interval = run.validation_interval
        self.show_interval = run.show_interval
        self.softmax = run.softmax
        self.hidden_layer_dimensions = run.hidden_layer_dimensions
        self.day_list = []
        self.run_nr = run_nr
        self.soft_label = soft_label
        self.soft_label_percent = soft_label_percent

        self.keep_probability_for_dropout = run.keep_probability_for_dropout




    def build_networks(self, number_of_networks=10, epochs=40, rank = 0):
        attributes_output = ["ret"]
        print("\nSTARTED PROCESS" + "\t\tProcessor #" + str(rank) + " stock #" + str(self.stock_nr) + "\t\tRun nr#" + str(self.run_nr))
        lftse100 = pi.InputPortolfioInformation(self.selectedFTSE100, attributes_output, self.fromDate, "LFTSE100wReturn.txt", 1,
                                                self.number_of_trading_days, normalize_method="minmax",
                                                one_hot_vector_interval=self.one_hot_vector_interval, is_output=True,
                                                start_time=self.start_time, rank=rank)
        if (lftse100.ended_up_being_to_many_NA_values == True):
            return None
        if (self.stock_nr == 62):
            return None


        # selectedSP500 = ssr.readSelectedStocks("TestInput.txt")
        # selectedFTSE100 = ssr.readSelectedStocks("TestOutput.txt")
        # sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, fromDate, "det-Input.txt", 7,
        #                                      number_of_trading_days, normalize_method="minmax", start_time=start_time)
        # lftse100 = pi.InputPortolfioInformation(selectedFTSE100, attributes_output, fromDate, "det-Output.txt", 1,
        #                                                 number_of_trading_days, normalize_method="minmax",
        #                                                 one_hot_vector_interval=one_hot_vector_interval, is_output=True,
        #                                                 start_time=start_time)
        case_generator = cg.CaseGenerator(self.sp500.normalized_portfolio_data, lftse100.portfolio_data,
                                          lftse100.normalized_portfolio_data, self.time_lags_sp, self.time_lags_ftse, self.one_hot_vector_interval,
                                          self.one_hot_size)
        cases = case_generator.cases
        fraction_of_cases_for_one_network = float(1.0 / float(number_of_networks))
        seperator0 = 0
        self.stock_result = res.StockResult(self.start_time, self.stock_nr)
        start_day_testing = 0
        for network_nr in range(0, number_of_networks):
            separator1 = int(round(len(cases) * fraction_of_cases_for_one_network)) + seperator0
            if (network_nr == number_of_networks - 1):
                separator1 = len(cases)
            case_manager = cm.CaseManager(cases[seperator0:separator1], self.time_lags_sp, validation_fraction=0.0,
                                          test_fraction=0.10, one_hot_vector_interval = self.one_hot_vector_interval, soft_label = self.soft_label, soft_label_percent=self.soft_label_percent)
            print("-----------Casemanager for stock nr: " + str(self.stock_nr) + " done---------------")
            start_day_testing += len(case_manager.get_training_cases()) + len(case_manager.get_validation_cases())

            end_day_testing = start_day_testing + len(case_manager.get_testing_cases())
            input_size = len(cases[0][0][0])
            output_size = len(cases[0][1][0])
            layer_dimension = [input_size]
            layer_dimension.extend(self.hidden_layer_dimensions)
            layer_dimension.append(output_size)

            neural_net = nn.NeuralNet(network_nr, layer_dimension, self.activation_functions, self.learning_rate,
                                      self.minibatch_size,
                                      self.time_lags_sp, self.cost_function, self.learning_method, case_manager,
                                      self.keep_probability_for_dropout,
                                      self.validation_interval, self.show_interval, self.softmax, self.start_time, rank)
            neural_net.run(epochs=epochs, sess=None, continued=None)

            self.stock_result.add_to_result(neural_net)
            for day in range(start_day_testing, end_day_testing):
                self.day_list.append(day)

            start_day_testing = end_day_testing
            seperator0 = separator1

        self.stock_result.generate_final_result_info(number_of_networks)
        print ("\nFINISHED PROCESS \tProcessor #" + str(rank) + " stock #" + str(self.stock_nr) + " in\t" +"%s seconds ---" % (time.time() - self.start_time))
        return self.stock_result
