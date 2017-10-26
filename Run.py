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

from mpi4py import MPI as MPI

#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

#Standarized names for learningn method:        "gradient_decent" - Gradient decent

class Run():

    def __init__(self, activation_functions, hidden_layer_dimension, time_lags, one_hot_vector_interval, number_of_networks, keep_probability_dropout,
                 from_date, number_of_trading_days, attributes_input, number_of_stocks,
                 learning_rate, minibatch_size, epochs):

        #Start timer
        self.start_time = time.time()
        self.end_time = time.time()

        #Network specific
        self.activation_functions = activation_functions        #["tanh", "tanh", "tanh", "tanh", "tanh", "sigmoid"]
        self.hidden_layer_dimensions = hidden_layer_dimension   #[100,50]
        self.time_lags = time_lags                              #3
        self.one_hot_vector_interval = one_hot_vector_interval  #[-0.000, 0.000]
        self.keep_probability_for_dropout = keep_probability_dropout #0.80
        self.number_of_networks = number_of_networks

        #Data set specific
        self.fromDate = from_date                               # "01.01.2008"
        self.number_of_trading_days = number_of_trading_days    #2000
        self.attributes_input = attributes_input                #["op", "cp"]
        self.selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
        self.number_of_stocks = number_of_stocks
        print("SPDRUSING")
        self.sp500 = pi.InputPortolfioInformation(self.selectedSP500, self.attributes_input, self.fromDate, "S&P500.txt", 7,
                                             self.number_of_trading_days, normalize_method="minmax", start_time=self.start_time)

        #Training specific
        self.learning_rate = learning_rate                      #0.1
        self.minibatch_size = minibatch_size                    #10
        self.cost_function = "cross_entropy"                    #TODO: vil vi bare teste denne?
        self.learning_method = "gradient_decent"                #TODO: er det progget inn andre loesninger?
        self.softmax = True
        self.epochs = epochs

        #Delete?
        self.one_hot_size = 3                                   #Brukes i NetworkManager
        self.initial_weight_range = [-1.0, 1.0]                 #TODO: fiks
        self.initial_bias_weight_range = [0.0, 0.0]             #TODO: fiks
        self.show_interval = None                               #Brukes i network manager
        self.validation_interval = None                         #Brukes i network manager

        #Results
        self.portfolio_day_up_returns = []      #Describes the return on every trade on long-strategy
        self.portfolio_day_down_returns = []    #Describes the return on every trade on short_strategy
        self.stock_results = []
        self.f = open("res.txt", "w");
        self.hyper_param_dict = None
        self.aggregate_counter_table = None




    def run_portfolio_in_parallell(self):
        from mpi4py import MPI as MPI
        comm = MPI.COMM_WORLD
        number_of_cores = comm.Get_size()  # Total number of processors
        rank = comm.Get_rank()
        if (rank == 0):
            selectedFTSE100 = self.generate_selected_list()
            number_of_stocks_to_test = self.number_of_stocks
            for prosessor_index in range(1, number_of_cores):
                end_of_range = (prosessor_index + 1) * int(number_of_stocks_to_test / number_of_cores)
                start_range = (prosessor_index) * int(number_of_stocks_to_test / number_of_cores)
                stock_information_for_processor = [start_range, end_of_range, copy.deepcopy(selectedFTSE100)]

                comm.send(stock_information_for_processor, dest=prosessor_index, tag=11)

                # for stock_nr in range(start_range, end_of_range):
                #     selectedFTSE100[stock_nr] = 1
                #     self.comm.send(nm.NetworkManager(self, selectedFTSE100, stock_nr), dest=prosessor_index, tag=11)
                #     selectedFTSE100[stock_nr] = 0
            for stock_nr in range(0, int(number_of_stocks_to_test / number_of_cores)):
                selectedFTSE100[stock_nr] = 1
                network_manager = nm.NetworkManager(self, selectedFTSE100, stock_nr)
                if (stock_nr == 0):
                    self.day_list = network_manager.day_list
                stock_result = network_manager.build_networks(number_of_networks=self.number_of_networks, epochs=self.epochs)

                self.stock_results.append(stock_result)
                selectedFTSE100[stock_nr] = 0


        else:
            stock_information_for_processor = comm.recv(source=0, tag=11)
            selectedFTSE100 = stock_information_for_processor[2]
            stock_results = []
            for stock_nr in range(stock_information_for_processor[0], stock_information_for_processor[1]):
                selectedFTSE100[stock_nr] = 1
                network_manager = nm.NetworkManager(self, selectedFTSE100, stock_nr)
                stock_result = network_manager.build_networks(number_of_networks=self.number_of_networks, epochs=self.epochs)
                stock_results.append(stock_result)
                selectedFTSE100[stock_nr] = 0

            comm.send(stock_results, dest=0, tag=11)  # Send result to master
            #network_manager = self.comm.recv(source=0, tag=11)
            #stock_result = network_manager.build_networks(number_of_networks=self.number_of_networks, epochs=self.epochs)
            #self.do_result_processing(stock_result)
            #self.comm.send(stock_result, dest=0, tag=11)  # Send result to master

        if (rank == 0):
            for i in range(1, number_of_cores):
                status = MPI.Status()
                recv_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                print("Got data: " + str(recv_data) + ", from processor: " + str(status.Get_source()))
                self.stock_results.extend(recv_data)

            hyp = self.generate_hyper_param_result()
            hyp_type_1 = [hyp[0], hyp[1]]
            hyp_type_2 = [hyp[2]]
            excel.ExcelFormatter(hyp_type_1, hyp_type_2, first_run=True) #prints to results.csv file
            #self.print_portfolio_return_graph()
            self.f.close()


    def generate_hyper_param_result(self):

        self.result_dict = {}

        self.result_dict["tot_return"] = self.get_total_return()
        self.result_dict["tot_day_std"] = self.get_total_day_std()


        self.result_dict["tot_long_return"] = self.get_portfolio_up_return()
        #self.result_dict["tot_day_long_std"] = self.get_day_long_std() #TODO: fix the day_long_std

        self.result_dict["tot_short_return"] = self.get_portfolio_down_return()
        #self.result_dict["tot_day_short_std"] = self.get_day_short_std() #TODO: fix the short_std

        self.aggregate_counter_table = self.get_aggregate_counter_table() # calculated here so it just have to be done once for precision and accuracy
        self.add_accuracy_to_result_dict()
        self.add_precision_to_result_dict()


        #The hyperparameters
        hyper_param_dict = self.generate_hyper_param_dict()

        stock_results_dict = {}
        #Per stock data:
        stock_results_dict["Stock_returns"] = self.generate_stock_return_list()
        stock_results_dict["Stock_accuracies"] = self.generate_stock_accuracies()
        stock_results_dict["Stock_long_return"] = self.generate_stock_long_returns()
        stock_results_dict["Stock_short_return"] = self.generate_stock_short_returns()

        res = [self.result_dict, hyper_param_dict, stock_results_dict]

        return res

    def generate_stock_long_returns(self):
        ret = []
        for stock_result in self.stock_results:
            ret.append(stock_result.get_tot_up_return())
        return ret

    def generate_stock_short_returns(self):
        ret = []
        for stock_result in self.stock_results:
            ret.append(stock_result.get_tot_down_return())
        return ret

    def generate_stock_return_list(self):
        stock_returns = []
        for stock_result in self.stock_results:
            stock_returns.append(stock_result.get_over_all_return())
        return stock_returns

    def generate_stock_accuracies(self):
        stock_accuracies = []
        for stock_result in self.stock_results:
            stock_accuracies.append(stock_result.get_total_pred_accuracy())
        return stock_accuracies

    def add_precision_to_result_dict(self):
        if(self.generate_hyper_param_dict() == None):
            self.aggregate_counter_table = self.get_aggregate_counter_table()
            up = 0
            s = 0
            d = 0
            for count in self.aggregate_counter_table["up"]:
                up += self.aggregate_counter_table["up"][count]
            for count in self.aggregate_counter_table["down"]:
                d += self.aggregate_counter_table["down"][count]
            for count in self.aggregate_counter_table["stay"]:
                s+= self.aggregate_counter_table["stay"][count]
        else:
            up = 0.0
            s = 0.0
            d = 0.0
            for count in self.aggregate_counter_table["up"]:
                up += self.aggregate_counter_table["up"][count]
            for count in self.aggregate_counter_table["down"]:
                d += self.aggregate_counter_table["down"][count]
            for count in self.aggregate_counter_table["stay"]:
                s+= self.aggregate_counter_table["stay"][count]
        if(up!=0):
            self.result_dict["prec_up_up"] = float(self.aggregate_counter_table["up"]["up"])/up
            self.result_dict["prec_up_d"] = float(self.aggregate_counter_table["up"]["down"])/up
            self.result_dict["prec_up_s"] = float(self.aggregate_counter_table["up"]["stay"])/up
        else:
            self.result_dict["prec_up_d"] = "N/A"
            self.result_dict["prec_up_s"] = "N/A"
            self.result_dict["prec_up_up"] = "N/A"

        if(d != 0):
            self.result_dict["prec_d_d"] = float(self.aggregate_counter_table["down"]["down"])/d
            self.result_dict["prec_d_s"] = float(self.aggregate_counter_table["down"]["stay"])/d
            self.result_dict["prec_d_up"] = float(self.aggregate_counter_table["down"]["up"])/d
        else:
            self.result_dict["prec_d_d"] = "N/A"
            self.result_dict["prec_d_s"] = "N/A"
            self.result_dict["prec_d_up"] ="N/A"

        if(s != 0):
            self.result_dict["prec_s_up"] = float(self.aggregate_counter_table["stay"]["up"])/s
            self.result_dict["prec_s_s"] = float(self.aggregate_counter_table["stay"]["stay"])/s
            self.result_dict["prec_s_d"] = float(self.aggregate_counter_table["stay"]["down"])/s
        else:
            self.result_dict["prec_s_d"] = "N/A"
            self.result_dict["prec_s_s"] = "N/A"
            self.result_dict["prec_s_up"] = "N/A"

    def add_accuracy_to_result_dict(self):
        up = 0
        s = 0
        d = 0
        for stock_result in self.stock_results:
            dict = stock_result.get_over_all_actual_map()
            up += dict["up"]
            s += dict["stay"]
            d += dict["down"]

        if(up!=0):
            self.result_dict["acc_up_up"] = float(self.aggregate_counter_table["up"]["up"])/up
            self.result_dict["acc_s_up"] = float(self.aggregate_counter_table["stay"]["up"])/up
            self.result_dict["acc_d_up"] = float(self.aggregate_counter_table["down"]["up"])/up
        else:
            self.result_dict["acc_d_up"] ="N/A"
            self.result_dict["acc_up_up"] = "N/A"
            self.result_dict["acc_s_up"] = "N/A"

        if(d != 0):
            self.result_dict["acc_d_d"] = float(self.aggregate_counter_table["down"]["down"])/d
            self.result_dict["acc_up_d"] = float(self.aggregate_counter_table["up"]["down"])/d
            self.result_dict["acc_s_d"] = float(self.aggregate_counter_table["stay"]["down"])/d
        else:
            self.result_dict["acc_up_d"] = "N/A"
            self.result_dict["acc_d_d"] = "N/A"
            self.result_dict["acc_s_d"] = "N/A"

        if(s != 0):
            self.result_dict["acc_up_s"] = float(self.aggregate_counter_table["up"]["stay"])/s
            self.result_dict["acc_d_s"] = float(self.aggregate_counter_table["down"]["stay"])/s
            self.result_dict["acc_s_s"] = float(self.aggregate_counter_table["stay"]["stay"])/s
        else:
            self.result_dict["acc_d_s"] = "N/A"
            self.result_dict["acc_s_s"] = "N/A"
            self.result_dict["acc_up_s"] = "N/A"



    def generate_hyper_param_dict(self):
        #"activation_functions", "hidden_layer_dimension", "time_lags", "one_hot_vector_interval", "keep_probability_dropout",
        #"from_date", "number_of_trading_days", "attributes_input",
        #"learning_rate", "minibatch_size")
        dict = {}
        dict["activation_functions"] = self.activation_functions
        dict["hidden_layer_dimension"] = self.hidden_layer_dimensions
        dict["time_lags"] = self.time_lags
        dict["one_hot_vector_interval"] = self.one_hot_vector_interval
        dict["keep_probability_dropout"] = self.keep_probability_for_dropout
        dict["from_date"] = self.fromDate
        dict["number_of_trading_days"] = self.number_of_trading_days
        dict["attr_input"] = self.attributes_input
        dict["learning_rate"] = self.learning_rate
        dict["minibatch_size"] = self.minibatch_size
        dict["number_of_stocks"] = self.number_of_stocks
        dict["epochs"] = self.epochs
        dict["#hidden_layers"] = len(self.hidden_layer_dimensions)
        dict["tot_test_size"] = self.stock_results[0].testing_sizes
        dict["#tr_case_pr_network"] = self.number_of_trading_days/self.number_of_networks
        return dict


    def get_aggregate_counter_table(self): #returns the dictionary with counts on [pred][actual] for keys ["up"]["up"] etc
        dictionary = {}
        dictionary["up"] = {}
        dictionary["up"]["up"] = 0
        dictionary["up"]["stay"] = 0
        dictionary["up"]["down"] = 0
        dictionary["stay"] = {}
        dictionary["stay"]["up"] = 0
        dictionary["stay"]["stay"] = 0
        dictionary["stay"]["down"] = 0
        dictionary["down"] = {}
        dictionary["down"]["up"] = 0
        dictionary["down"]["stay"] = 0
        dictionary["down"]["down"] = 0

        for stock_result in self.stock_results:
            dictionaries = stock_result.get_counter_dictionaries()
            for dict in dictionaries:
                dictionary["up"]["up"] += dict["up"]["up"]
                dictionary["up"]["stay"] +=dict["up"]["stay"]
                dictionary["up"]["down"] += dict["up"]["down"]

                dictionary["stay"]["up"] += dict["stay"]["up"]
                dictionary["stay"]["stay"] += dict["stay"]["stay"]
                dictionary["stay"]["down"] += dict["stay"]["down"]

                dictionary["down"]["up"] += dict["down"]["up"]
                dictionary["down"]["stay"] += dict["down"]["stay"]
                dictionary["down"]["down"] += dict["down"]["down"]

        return dictionary



    def write_all_results_to_file(self):
        for stock_result in self.stock_results:
            result_string = stock_result.genereate_result_string()
            print(result_string)
            self.write_result_to_file(result_string, stock_result.stock_nr)

    def run_portfolio(self):
        self.f = open("res.txt", "w")
        selectedFTSE100 = self.generate_selected_list()
        testing_size = 0
        number_of_stocks_to_test = self.number_of_stocks
        #array with all the StockResult objects
        for stock_nr in range(0, number_of_stocks_to_test):
            selectedFTSE100[stock_nr] = 1
            network_manager = nm.NetworkManager(self, selectedFTSE100, stock_nr)
            stock_result = network_manager.build_networks(number_of_networks=self.number_of_networks, epochs=self.epochs)
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
        self.end_time = time.time()

    def get_total_return(self):
        portolfio_day_returns = self.find_portfolio_day_to_day_return(self.stock_results)
        portfolio_day_returns_as_percentage = self.make_return_percentage(portolfio_day_returns)
        tot_return = float(portfolio_day_returns_as_percentage[-1])
        return tot_return

    def get_total_day_std(self):
        portfolio_day_returns = self.find_portfolio_day_to_day_return(self.stock_results)
        standard_deviation_of_returns = np.std(self.convert_accumulated_portfolio_return_to_day_returns(portfolio_day_returns))
        return standard_deviation_of_returns

    def print_portfolio_return_graph(self):
        if (len(self.stock_results) > 0):
            portolfio_day_returns = self.find_portfolio_day_to_day_return(self.stock_results)
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

    def get_day_long_std(self):
        up_std = np.std(self.portfolio_day_up_returns)
        return up_std

    def get_day_short_std(self):
        down_std = np.std(self.portfolio_day_down_returns)
        return down_std

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

    def find_portfolio_day_to_day_return(self, stock_results):
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
        for day in range(0, len(stock_results[0].get_day_down_returns())):
            total_day_ret = 0
            for stock_result in stock_results:
                total_day_ret += stock_result.get_day_down_returns()[day]
            day_avg = total_day_ret/float(len(stock_results))
            self.portfolio_day_down_returns.append(day_avg)

    def collect_portfolio_day_up_returns(self, stock_results): #Does not return anything
        for day in range(0, len(stock_results[0].get_day_up_returns())):
            total_day_ret = 0
            for stock_result in stock_results:
                total_day_ret += stock_result.get_day_up_returns()[day]
            day_avg = total_day_ret/float(len(stock_results))
            self.portfolio_day_up_returns.append(day_avg)

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


