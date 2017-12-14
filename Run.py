import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import copy
import os
os.environ['T' \
           'F_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import StockResult as res
import NetworkManager as nm
import matplotlib.pyplot as plt
import numpy as np
import ExcelFormatter as excel
import HyperParamResult as hpr
import GoogleSheetWriter as gsw

from mpi4py import MPI as MPI
from time import gmtime, strftime

#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

#Standarized names for learningn method:        "gradient_decent" - Gradient decent

class Run():

    def __init__(self, activation_functions, hidden_layer_dimension, time_lags_sp, time_lags_ftse, one_hot_vector_interval, number_of_networks, keep_probability_dropout,
                 from_date, number_of_trading_days, attributes_input, number_of_stocks,
                 learning_rate, minibatch_size, epochs, rf_rate, run_nr, sp500, soft_label, soft_label_percent, run_description):

        self.run_description = run_description

        #Start timer
        self.start_time = time.time()
        self.end_time = time.time()
        self.run_nr = run_nr

        self.rf_rate = rf_rate

        #Network specific
        self.activation_functions = activation_functions        #["tanh", "tanh", "tanh", "tanh", "tanh", "sigmoid"]
        self.hidden_layer_dimensions = hidden_layer_dimension   #[100,50]
        self.time_lags_sp = time_lags_sp                              #3
        self.time_lags_ftse = time_lags_ftse                          #3
        self.one_hot_vector_interval = one_hot_vector_interval  #[-0.000, 0.000]
        self.keep_probability_for_dropout = keep_probability_dropout #0.80
        self.number_of_networks = number_of_networks

        #Data set specific
        self.fromDate = from_date                               # "01.01.2008"
        self.number_of_trading_days = number_of_trading_days    #2000
        self.attributes_input = attributes_input                #["op", "cp"]
        #self.selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
        self.number_of_stocks = number_of_stocks
        #self.sp500 = pi.InputPortolfioInformation(self.selectedSP500, self.attributes_input, self.fromDate, "S&P500.txt", 7,
        #                                     self.number_of_trading_days, normalize_method="minmax", start_time=self.start_time)
        self.sp500 = sp500
        #Training specific
        self.learning_rate = learning_rate                      #0.1
        self.minibatch_size = minibatch_size                    #10
        self.cost_function = "cross_entropy"                    #TODO: vil vi bare teste denne?
        self.learning_method = "gradient_decent"                #TODO: er det progget inn andre loesninger?
        self.softmax = True
        self.epochs = epochs
        self.soft_label = soft_label
        self.soft_label_percent = soft_label_percent

        #Delete?
        self.one_hot_size = 3                                   #Brukes i NetworkManager
        self.initial_weight_range = [-1.0, 1.0]                 #TODO: fiks
        self.initial_bias_weight_range = [0.0, 0.0]             #TODO: fiks
        self.show_interval = None                               #Brukes i network manager
        self.validation_interval = 1                         #Brukes i network manager

        #Results
        self.portfolio_day_up_returns = []      #Describes the return on every trade on long-strategy
        self.portfolio_day_down_returns = []    #Describes the return on every trade on short_strategy
        self.stock_results = []

        self.hyper_param_dict = None
        self.aggregate_counter_table = None

        #Map from stock nr to stock
        self.global_stock_results = {}



    def run_portfolio_in_parallell(self):
        from mpi4py import MPI as MPI
        comm = MPI.COMM_WORLD
        number_of_cores = comm.Get_size()  # Total number of processors
        rank = comm.Get_rank()
        if (rank == 0):
            print("\n\n\n------------------------------ RUN NR " + str(self.run_nr) + " ------------------------------")
            selectedFTSE100 = self.generate_selected_list()
            delegated = self.delegate_stock_nr(number_of_cores, self.number_of_stocks)

            for prosessor_index in range(1, number_of_cores):
                stock_information_for_processor = [delegated[prosessor_index], copy.deepcopy(selectedFTSE100)]
                comm.send(stock_information_for_processor, dest=prosessor_index, tag=11)

            for stock_nr in delegated[0]:
                selectedFTSE100[stock_nr] = 1
                self.generate_network_manager(copy.deepcopy(selectedFTSE100), stock_nr, rank)
                selectedFTSE100[stock_nr] = 0

        else:
            stock_information_for_processor = comm.recv(source=0, tag=11)
            selected = stock_information_for_processor[1]
            stock_results = []
            for stock_nr in stock_information_for_processor[0]: #TODO: potentially conflicting when less stocks than processors
                selected[stock_nr] = 1
                rank = comm.Get_rank()
                stock_result = self.generate_network_manager(selected, stock_nr, rank)
                if (stock_result != None):
                    stock_results.append(stock_result)
                selected[stock_nr] = 0

            comm.send(stock_results, dest=0, tag=11)  # Send result to master


        if (rank == 0):
            for i in range(1, number_of_cores):
                status = MPI.Status()
                recv_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                print("COMPLETE! \t\tProcessor #" + str(status.Get_source()) + "\t" +"%s seconds ---" % (time.time() - self.start_time) + "\t run nr: "+ str(self.run_nr))
                self.stock_results.extend(recv_data)

            hyp, ordered_label_list_type_1, ordered_label_list_type_2 = self.generate_hyper_param_result()
            hyp_type_1 = [hyp[0], hyp[1]]
            hyp_type_2 = [hyp[2]]
            #excel.ExcelFormatter(hyp_type_1, hyp_type_2, ordered_label_list_type_1, line_number=self.run_nr) #prints to results.csv file
            #gsw.main(hyp_type_1, hyp_type_2, ordered_label_list_type_1, ordered_label_list_type_2)
            #self.print_portfolio_return_graph()

    def generate_network_manager(self, selected, stock_nr, rank):
        network_manager = nm.NetworkManager(self, selected, stock_nr, self.run_nr, self.rf_rate, self.soft_label, self.soft_label_percent)
        stock_result = network_manager.build_networks(number_of_networks=self.number_of_networks, epochs=self.epochs,
                                                      rank=rank)
        self.add_to_stock_results(stock_result, network_manager)
        return stock_result

    def add_to_stock_results(self, stock_result, network_manager):
        if (stock_result != None):
            self.stock_results.append(stock_result)
            self.day_list = network_manager.day_list

    def generate_hyper_param_result(self):

        ordered_label_list_for_hyp_type_1 = []

        # The hyperparameters
        hyper_param_dict = self.generate_hyper_param_dict(ordered_label_list_for_hyp_type_1)

        #The results
        self.result_dict = {}
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_training_acc", self.generate_tot_training_accuracy()))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_test_acc", self.generate_tot_accuracy()))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_prec", self.generate_tot_precision()))

        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_return", self.get_total_return()))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_day_std", self.get_total_day_std()))

        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_long_return", (self.get_portfolio_up_return()-1)))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_day_long_std", self.get_day_long_std()))

        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_short_return", self.get_portfolio_down_return() - 1))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "tot_day_short_std", self.get_day_short_std()))

        sharpe_ratios = self.compute_sharpe_ratios()
        self.result_dict["sharpe_tot_ratio"] = sharpe_ratios[0] #TODO
        self.result_dict["sharpe_short_ratio"] = sharpe_ratios[1] #TODO
        self.result_dict["sharpe_long_ratio"] = sharpe_ratios[2] #TODO

        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "sharpe_tot_ratio", sharpe_ratios[0]))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "sharpe_short_ratio", sharpe_ratios[1]))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "sharpe_long_ratio", sharpe_ratios[2]))

        #The accuracy and precision
        estimated_map =  self.generate_total_estimated()
        actual_map =  self.generate_total_actual()
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "estimated_up", estimated_map["up"]))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "actual_up", actual_map["up"]))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "estimated_stay", estimated_map["stay"]))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "actual_stay", actual_map["stay"]))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "estimated_down", estimated_map["down"]))
        ordered_label_list_for_hyp_type_1.append(
            self.define_key_and_put_in_dict(self.result_dict, "actual_down", actual_map["down"]))
        self.aggregate_counter_table = self.get_aggregate_counter_table() # calculated here so it just have to be done once for precision and accuracy
        self.add_accuracy_to_result_dict(ordered_label_list_for_hyp_type_1)
        self.add_precision_to_result_dict(ordered_label_list_for_hyp_type_1)

        stock_results_dict = {}
        ordered_label_list_for_hyp_type_2 = []
        ordered_label_list_for_hyp_type_2.append(
            self.define_key_and_put_in_dict(stock_results_dict, "Stock_accuracies", self.generate_stock_accuracies()))
        ordered_label_list_for_hyp_type_2.append(
            self.define_key_and_put_in_dict(stock_results_dict, "Stock_returns",
                                            self.generate_stock_return_list()))
        ordered_label_list_for_hyp_type_2.append(
            self.define_key_and_put_in_dict(stock_results_dict, "Stock_sd",
                                            self.generate_stock_sd_list()))
        ordered_label_list_for_hyp_type_2.append(
            self.define_key_and_put_in_dict(stock_results_dict, "Stock_sharpe_ratio",
                                            self.generate_stock_sharpe_ratio_list()))
        ordered_label_list_for_hyp_type_2.append(
            self.define_key_and_put_in_dict(stock_results_dict, "Stock_long_return",
                                            self.generate_stock_long_returns()))
        ordered_label_list_for_hyp_type_2.append(
            self.define_key_and_put_in_dict(stock_results_dict, "Stock_short_return",
                                            self.generate_stock_short_returns()))


        portfolio_dict = {}
        ordered_label_list_for_hyp_type_2.append(
            self.define_key_and_put_in_dict(portfolio_dict, "portfolio_accumulated_day_ret",
                                            self.find_portfolio_day_to_day_accumulated_return(self.stock_results)))
        #portfolio_dict["portfolio_accumulated_day_ret"] = self.find_portfolio_day_to_day_accumulated_return(self.stock_results)

        res = [self.result_dict, hyper_param_dict, stock_results_dict, portfolio_dict]

        return res, ordered_label_list_for_hyp_type_1, ordered_label_list_for_hyp_type_2

    def define_key_and_put_in_dict(self, dict, key, value):
        dict[key] = value
        return key

    def generate_total_estimated(self):
        total_estimated_map = {}
        for classification in self.stock_results[0].over_all_estimated_map:
            tot_estimated_number = 0
            for stock_result in self.stock_results:
                tot_estimated_number += stock_result.over_all_estimated_map[classification]
                total_estimated_map[classification] = tot_estimated_number
        return total_estimated_map

    def generate_total_actual(self):
        total_actual_map = {}
        for classification in self.stock_results[0].over_all_actual_map:
            tot_actual_number = 0
            for stock_result in self.stock_results:
                tot_actual_number += stock_result.over_all_actual_map[classification]
                total_actual_map[classification] = tot_actual_number
        return total_actual_map

    def generate_tot_accuracy(self):
        total_acc = 0
        for stock_result in self.stock_results:
            total_acc += stock_result.accuracy
            # stock_result.accuracy == 1.0 and stock_result.over_all_return == 1.0
            if (stock_result.get_total_pred_accuracy() == 1.0 and stock_result.get_over_all_return() == 1.0):
                raise ValueError("Most likely something wrong here! Accuracy and return is 1.0.")
                print("THE WRONG SHIT IS FOUND:")
                predicted = stock_result.predictions
                targets = stock_result.targets
                actual_returns = stock_result.actual_returns
                print("Tar\tEst\tRet")
                for i in range(0, len(targets)):
                    print(str(targets[i])+"\t"+str(predicted[i])+"\t"+str(actual_returns[i]))

        return total_acc/float(len(self.stock_results))

    def generate_tot_training_accuracy(self):
        tot_acc = 0
        for stock_result in self.stock_results:
            tot_acc += stock_result.tot_traning_acc
        return tot_acc/float(len(self.stock_results))

    def generate_tot_precision(self):
        total_prec = 0
        for stock_result in self.stock_results:
            total_prec += stock_result.over_all_precision
        return total_prec/float(len(self.stock_results))

    def delegate_stock_nr(self, processors, nr_of_stocks):
        delegated = []
        for n in range(processors):
            delegated.append([])

        for nr in range(0, nr_of_stocks):
            index = nr % processors
            delegated[index].append(nr)
        return delegated

    def generate_stock_long_returns(self):
        ret = []
        for stock_result in self.stock_results:
            ret.append([stock_result.get_tot_up_return(), stock_result.stock_nr])
        return ret

    def generate_stock_short_returns(self):
        ret = []
        for stock_result in self.stock_results:
            ret.append([stock_result.get_tot_down_return(), stock_result.stock_nr])
        return ret

    def generate_stock_return_list(self):
        stock_returns = []
        for stock_result in self.stock_results:
            stock_returns.append([stock_result.get_over_all_return(), stock_result.stock_nr])
        return stock_returns

    def generate_stock_sd_list(self):
        stock_sds = []
        for stock_result in self.stock_results:
            stock_sds.append([stock_result.over_all_standard_deviation_on_day_return, stock_result.stock_nr])
        return stock_sds

    def generate_stock_sharpe_ratio_list(self):
        stock_sharpe_ratios = []
        for stock_result in self.stock_results:
            stock_sharpe_ratios.append([stock_result.over_all_sharp_ratio, stock_result.stock_nr])
        return stock_sharpe_ratios

    def generate_stock_accuracies(self):
        stock_accuracies = []
        for stock_result in self.stock_results:
            stock_accuracies.append([stock_result.get_total_pred_accuracy(), stock_result.stock_nr])
        return stock_accuracies

    def write_all_results_to_file(self):
        for stock_result in self.stock_results:
            result_string = stock_result.genereate_result_string()
            print(result_string)
            self.write_result_to_file(result_string, stock_result.stock_nr)

    # def run_portfolio(self):
    #     self.f = open("res.txt", "w")
    #     selectedFTSE100 = self.generate_selected_list()
    #     testing_size = 0
    #     number_of_stocks_to_test = self.number_of_stocks
    #     #array with all the StockResult objects
    #     for stock_nr in range(0, number_of_stocks_to_test):
    #         selectedFTSE100[stock_nr] = 1
    #         network_manager = nm.NetworkManager(self, selectedFTSE100, stock_nr)
    #         stock_result = network_manager.build_networks(number_of_networks=self.number_of_networks, epochs=self.epochs)
    #         result_string = stock_result.genereate_result_string()
    #
    #         self.stock_results.append(stock_result)
    #
    #         if (stock_nr == 0):
    #             self.day_list = network_manager.day_list
    #         print(result_string)
    #         self.write_result_to_file(result_string, stock_nr)
    #         selectedFTSE100[stock_nr] = 0
    #
    #     self.write_long_results()
    #     self.write_short_results()
    #     self.print_portfolio_return_graph()
    #     self.f.close()
    #     self.end_time = time.time()

    def get_total_return(self):
        portolfio_day_returns = self.find_portfolio_day_to_day_accumulated_return(self.stock_results)
        portfolio_day_returns_as_percentage = self.make_return_percentage(portolfio_day_returns)
        tot_return = float(portfolio_day_returns_as_percentage[-1]/100)
        return (portolfio_day_returns[-1]/100)+1

    def get_total_day_std(self):
        portfolio_day_returns = self.find_portfolio_day_to_day_accumulated_return(self.stock_results)
        standard_deviation_of_returns = np.std(self.convert_accumulated_portfolio_return_to_day_returns(portfolio_day_returns))
        return standard_deviation_of_returns

    def print_portfolio_return_graph(self):
        if (len(self.stock_results) > 0):
            portolfio_day_returns = self.find_portfolio_day_to_day_accumulated_return(self.stock_results)
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

    def find_portfolio_day_to_day_accumulated_return(self, stock_results):
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

    def compute_sharpe_ratios(self):
        sharpe_ratios = [0,0,0] #0 - tot, 1 - short, 2 - long
        sharpe_ratios[0] = (self.get_total_return()-self.rf_rate)/self.get_total_day_std()
        sharpe_ratios[1] = (self.get_portfolio_down_return()-self.rf_rate)/self.get_day_short_std()
        sharpe_ratios[2] = (self.get_portfolio_up_return()-self.rf_rate)/self.get_day_long_std()
        return sharpe_ratios

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


    def add_precision_to_result_dict(self, ordered_label_list):
        ordered_label_list.append("prec_up_up")
        ordered_label_list.append("prec_d_d")
        ordered_label_list.append("prec_s_s")
        ordered_label_list.append("prec_up_d")
        ordered_label_list.append("prec_up_s")
        ordered_label_list.append("prec_d_s")
        ordered_label_list.append("prec_d_up")
        ordered_label_list.append("prec_s_d")
        ordered_label_list.append("prec_s_up")
        if(self.aggregate_counter_table == None):
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

    def add_accuracy_to_result_dict(self, ordered_label_list):
        up = 0
        s = 0
        d = 0
        ordered_label_list.append("acc_up_up")
        ordered_label_list.append("acc_d_d")
        ordered_label_list.append("acc_s_s")
        ordered_label_list.append("acc_d_up")
        ordered_label_list.append("acc_s_up")
        ordered_label_list.append("acc_up_d")
        ordered_label_list.append("acc_s_d")
        ordered_label_list.append("acc_up_s")
        ordered_label_list.append("acc_s_d")
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



    def generate_hyper_param_dict(self, ordered_label_list):
        #"activation_functions", "hidden_layer_dimension", "time_lags", "one_hot_vector_interval", "keep_probability_dropout",
        #"from_date", "number_of_trading_days", "attributes_input",
        #"learning_rate", "minibatch_size")
        dict = {}
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "time", strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "run_description", self.run_description))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "activation_functions", self.activation_functions))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "hidden_layer_dimension", self.hidden_layer_dimensions))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "time_lags_sp", self.time_lags_sp))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "time_lags_ftse", self.time_lags_ftse))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "softlabel", self.soft_label))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "one_hot_vector_interval", self.one_hot_vector_interval))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "keep_probability_dropout", self.keep_probability_for_dropout))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "from_date", self.fromDate))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "number_of_trading_days", self.number_of_trading_days))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "attr_input", self.attributes_input))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "learning_rate", self.learning_rate))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "minibatch_size", self.minibatch_size))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "number_of_stocks", self.number_of_stocks))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "epochs", self.epochs))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "#hidden_layers", len(self.hidden_layer_dimensions)))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "tot_test_size", self.stock_results[0].testing_sizes))
        ordered_label_list.append(self.define_key_and_put_in_dict(dict, "#tr_case_pr_network", self.number_of_trading_days/self.number_of_networks))
        # dict["activation_functions"] = self.activation_functions
        # dict["hidden_layer_dimension"] = self.hidden_layer_dimensions
        # dict["time_lags"] = self.time_lags
        # dict["one_hot_vector_interval"] = self.one_hot_vector_interval
        # dict["keep_probability_dropout"] = self.keep_probability_for_dropout
        # dict["from_date"] = self.fromDate
        # dict["number_of_trading_days"] = self.number_of_trading_days
        # dict["attr_input"] = self.attributes_input
        # dict["learning_rate"] = self.learning_rate
        # dict["minibatch_size"] = self.minibatch_size
        # dict["number_of_stocks"] = self.number_of_stocks
        # dict["epochs"] = self.epochs
        # dict["#hidden_layers"] = len(self.hidden_layer_dimensions)
        # dict["tot_test_size"] = self.stock_results[0].testing_sizes
        # dict["#tr_case_pr_network"] = self.number_of_trading_days/self.number_of_networks
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

