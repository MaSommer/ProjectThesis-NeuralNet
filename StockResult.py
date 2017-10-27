import numpy as np
import time
import Stringformatter as string_formatter


class StockResult():

    def __init__(self, start_time, stock_nr):
        self.stock_nr = stock_nr        #TODO: should be changed to stock name
        self.accuracies = []            #accuracies per period
        self.accuracy_info_list = []    # list of 3x3 dict with accuracy for different pairs of prediction_actualValue, for every net
        self.precision_info_list = []   #list of 3x3 dict for every net with hit percentage on different prediction_value pairs u|u = [1][1]
        self.over_all_returns = []      #return per neural net (period with test data). total over each period
        self.up_returns = []            #returns form doing a short-only strategy
        self.down_returns = []          #returns from doing a buy-only strategy
        self.correct_pred_beg_streak_list = []
        self.actual_map_list = []       #list with target dicts for each network.
        self.estimated_map_list = []
        self.returns_up  = []
        self.returns_down = []

        self.day_down_returns = []
        self.day_up_returns = []

        self.day_returns_list = []      #accumulated returns from day to day investment.
        self.counter_dictionaries = []  #prediction_value pairs that maps the counts of each happening. uu, dd ss etc. list for each network
                                        # uu = counter_dictionaries["up"]["up"], other - "stay", "down"

        self.estimated_map_list = []
        self.testing_sizes = []
        self.total_accuracy_sum = 0.0
        self.total_testing_cases = 0.0
        self.total_precision_sum = 0.0
        self.start_time = start_time


    def get_counter_dictionaries(self):
        return self.counter_dictionaries

    def get_stock_name(self):
        return self.stock_nr

    def get_pred_accuracies(self):
        return self.accuracies

    def get_total_pred_accuracy(self):
        return self.accuracy

    def get_number_of_days(self):
        return len(self.day_returns_list)

    def get_actual_map_list(self):
        return self.actual_map_list

    def get_estimated_map_list(self):
        return self.estimated_map_list

    def get_periodic_returns(self):
        return self.over_all_returns

    def get_over_all_return(self):
        return self.over_all_return

    def get_tot_up_return(self):
        return self.tot_up_return

    def get_day_up_returns(self):
        return self.day_up_returns

    def get_day_down_returns(self):
        return self.day_down_returns

    def get_tot_down_return(self):
        return self.tot_down_return

    def get_periodic_down_returns(self):
        return self.down_returns

    def get_periodic_up_returns(self):
        return self.up_returns

    def get_over_all_actual_map(self):
        return self.over_all_actual_map

    def generate_final_result_info(self, number_of_networks):
        self.accuracy = self.total_accuracy_sum / self.total_testing_cases
        self.total_acc_info = self.generate_total_info(number_of_networks, self.accuracy_info_list)
        self.total_precision_info = self.generate_total_info(number_of_networks, self.precision_info_list)
        self.over_all_correct_pred_beg_streak_avg = sum(self.correct_pred_beg_streak_list) / len(self.correct_pred_beg_streak_list)

        #Generating return info on up,down and total
        self.over_all_return = self.generate_over_all_return(self.over_all_returns)
        self.tot_down_return = self.generate_down_return(self.down_returns)
        self.tot_up_return = self.generate_up_return(self.up_returns)

        self.over_all_actual_map = self.generate_over_all_actual_and_estimated_map(self.actual_map_list)
        self.over_all_estimated_map = self.generate_over_all_actual_and_estimated_map(self.estimated_map_list)

        self.over_all_precision = self.generate_over_all_precision()
        print("\n\n")

    def generate_over_all_actual_and_estimated_map(self, map_list):
        total_numbers = self.feed_actual_estimated_map()
        for classification in map_list[0]:
            for map in map_list:
                info = map[classification]
                total_numbers[classification] += info
        return total_numbers

    def generate_total_info(self, number_of_networks, info_list):
        total_info = self.feed_accuracy_relevant_dictionaries()
        total_numbers = self.feed_accuracy_relevant_dictionaries()
        for prediction in info_list[0]:
            for target in info_list[0][prediction]:
                for acc_info in info_list:
                    info = acc_info[prediction][target]
                    total_numbers[prediction][target] += info
        for prediction in info_list[0]:
            for target in info_list[0][prediction]:
                total_info[prediction][target] = float(total_numbers[prediction][target]) / float(number_of_networks)
        return total_info

    def generate_over_all_precision(self):
        tot_predicted = 0
        for classification in self.over_all_estimated_map:
            tot_predicted += self.over_all_estimated_map[classification]
        weights = {}
        for classification in self.over_all_estimated_map:
            weights[classification] = self.over_all_estimated_map[classification]/tot_predicted
        prec = {}
        for classification in self.total_precision_info:
            prec[classification] = self.total_precision_info[classification][classification]
        tot_prec = 0
        for classification in weights:
            tot_prec += prec[classification]*weights[classification]
        return tot_prec


    def generate_over_all_return(self, over_all_returns):
        tot_ret = 1.0
        for ret in over_all_returns:
            tot_ret *= ret
        return tot_ret

    def generate_up_return(self, up_returns):
        up_ret = 1.0
        for ret in up_returns:
            up_ret *= ret
        return up_ret

    def generate_down_return(self, down_returns):
        down_ret = 1.0
        for ret in down_returns:
            down_ret *= ret
        return down_ret

    def feed_actual_estimated_map(self):
        dictionary = {}
        dictionary["up"] = 0
        dictionary["stay"] = 0
        dictionary["down"] = 0
        return dictionary

    def feed_accuracy_relevant_dictionaries(self):
        dictionary = {}
        dictionary["up"] = {}
        # the list is [pred, target] counts how many predicted up and also target up
        dictionary["up"]["up"] = 0.0
        dictionary["up"]["stay"] = 0.0
        dictionary["up"]["down"] = 0.0
        dictionary["stay"] = {}
        dictionary["stay"]["up"] = 0.0
        dictionary["stay"]["stay"] = 0.0
        dictionary["stay"]["down"] = 0.0
        dictionary["down"] = {}
        dictionary["down"]["up"] = 0.0
        dictionary["down"]["stay"] = 0.0
        dictionary["down"]["down"] = 0.0
        return dictionary

    def add_day_up_returns(self, neural_net):
        for ret in neural_net.results.get_day_up_returns():
            self.day_up_returns.append(ret)

    def add_day_down_returns(self, neural_net):
        for ret in neural_net.results.get_day_down_returns():
            self.day_down_returns.append(ret)

    def add_to_result(self, neural_net):
        self.accuracies.append(neural_net.accuracy)
        self.accuracy_info_list.append(neural_net.results.accuracy_information)
        self.precision_info_list.append(neural_net.results.precision_information)

        self.over_all_returns.append(neural_net.results.overall_return)
        self.up_returns.append(neural_net.results.get_up_return())
        self.down_returns.append(neural_net.results.get_down_return())

        #Adding day to day returns on each strategy. Collects from the neural_network_result class
        self.add_day_down_returns(neural_net)
        self.add_day_up_returns(neural_net)

        self.correct_pred_beg_streak_list.append((neural_net.results.number_of_correct_predication_beginning_streak))
        self.estimated_map_list.append(neural_net.results.estimated_map)
        self.actual_map_list.append(neural_net.results.actual_map)
        self.counter_dictionaries.append(neural_net.results.counter_dict)

        self.update_day_returns(neural_net.results.day_returns)
        self.testing_sizes.append(neural_net.testing_size)

        self.total_accuracy_sum += neural_net.accuracy * neural_net.testing_size
        self.total_testing_cases += neural_net.testing_size

    def update_day_down_returns(self, day_down_returns):
        current_return = 1.0
        if(len(self.day_down_returns)>0):
            current_return = self.day_down_returns_list[-1]
        for ret in day_down_returns:
            ret*=current_return
            self.day_down_returns_list.append(ret)

    def update_day_returns(self, day_returns):
        current_return = 1.0
        if (len(self.day_returns_list) > 0):
            current_return = self.day_returns_list[-1]
        for ret in day_returns:
            ret *= current_return
            self.day_returns_list.append(ret)

    def print_accuracy_info(self):
        print("Running time: " + "\t %s seconds ---" % (time.time() - self.start_time))
        print("--- Over all accuracy: " + str(self.accuracy) + " ---")
        print("--- Over all accuracies: " + str(self.accuracies) + " ---")
        self.print_sequence("accuracy categorization", self.total_acc_info, "rate", self.over_all_actual_map, "actual")
        self.print_sequence("precision", self.total_precision_info, "precision", self.over_all_estimated_map, "estimated")
        print("--- Over all returns: " + str(self.over_all_returns))
        print("--- Total return: " + "{0:.4f}%".format((self.over_all_return - 1) * 100))
        print("--- Over all correct prediction beginning streak: " + str(self.correct_pred_beg_streak_list))
        print("--- Over all correct prediction beginning streak avg: " + str(self.over_all_correct_pred_beg_streak_avg))

        print("\n\n")


    def print_sequence(self, name, map_to_print, spesification ,estimated_actual, spec_name):
        print("--- Over all accuracy " + name + ": ")
        for classification in estimated_actual:
            print("\t--- Total number of " + spec_name + " " + classification + ": " + str(estimated_actual[classification]))
        for prediction in map_to_print:
            for target in map_to_print[prediction]:
                print("\t\t--- " + prediction + "-" + target + "-" + spesification + ": " + str(
                    map_to_print[prediction][target]))

    def genereate_result_string(self):
        result = ""
        result += "Running time: " + "\t %s seconds ---" % (time.time() - self.start_time) + "\n"
        result +=  "--- Over all accuracy: " + str(self.accuracy) + " ---\n"
        result +=  "--- Over all accuracies: " + str(self.accuracies) + " ---\n"
        result += self.generate_sequence("accuracy categorization", self.total_acc_info, "rate", self.over_all_actual_map, "actual")
        result += self.generate_sequence("precision", self.total_precision_info, "precision", self.over_all_estimated_map, "estimated")
        result += "--- Over all returns: " + str(self.over_all_returns) + "\n"
        result += "--- Total return: " + "{0:.4f}%".format((self.over_all_return - 1) * 100) + "\n"
        result += "--- Over all correct prediction beginning streak: " + str(self.correct_pred_beg_streak_list) + "\n"
        result += "--- Over all correct prediction beginning streak avg: " + str(self.over_all_correct_pred_beg_streak_avg) + "\n"
        sf = string_formatter.Stringformatter()
        return sf.generate_stock_result_string(self)

    def generate_sequence(self, name, map_to_print, spesification ,estimated_actual, spec_name):
        sequence = ""
        sequence += "--- Over all " + name + ": \n"
        for classification in estimated_actual:
            sequence +="\t--- Total number of " + spec_name + " " + classification + ": " + str(estimated_actual[classification]) + "\n"
        for prediction in map_to_print:
            for target in map_to_print[prediction]:
               sequence += "\t\t--- " + prediction + "-" + target + "-" + spesification + ": " + str(
                    map_to_print[prediction][target]) + "\n"
        return sequence