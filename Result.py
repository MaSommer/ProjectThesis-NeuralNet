import numpy as np
import time


class Result():

    def __init__(self, start_time):
        self.accuracies = []
        self.accuracy_info_list = []
        self.precision_info_list = []
        self.over_all_returns = []
        self.correct_pred_beg_streak_list = []
        self.actual_map_list = []
        self.estimated_map_list = []
        self.total_accuracy_sum = 0.0
        self.total_testing_cases = 0.0
        self.start_time = start_time

    def generate_final_result_info(self, number_of_networks):
        self.accuracy = self.total_accuracy_sum / self.total_testing_cases
        self.total_acc_info = self.generate_total_info(number_of_networks, self.accuracy_info_list)
        self.total_precision_info = self.generate_total_info(number_of_networks, self.precision_info_list)
        self.over_all_correct_pred_beg_streak_avg = sum(self.correct_pred_beg_streak_list) / len(self.correct_pred_beg_streak_list)
        self.over_all_return = self.generate_over_all_return(self.over_all_returns)
        self.over_all_actual_map = self.generate_over_all_actual_and_estimated_map(self.actual_map_list)
        self.over_all_estimated_map = self.generate_over_all_actual_and_estimated_map(self.estimated_map_list)

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

    def generate_over_all_return(self, over_all_returns):
        tot_ret = 1.0
        for ret in over_all_returns:
            tot_ret *= ret
        return tot_ret


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

    def add_to_result(self, neural_net):
        self.accuracies.append(neural_net.accuracy)
        self.accuracy_info_list.append(neural_net.results.accuracy_information)
        self.precision_info_list.append(neural_net.results.precision_information)
        self.over_all_returns.append(neural_net.results.overall_return)
        self.correct_pred_beg_streak_list.append((neural_net.results.number_of_correct_predication_beginning_streak))
        self.estimated_map_list.append(neural_net.results.estimated_map)
        self.actual_map_list.append(neural_net.results.actual_map)
        self.total_accuracy_sum += neural_net.accuracy * neural_net.testing_size
        self.total_testing_cases += neural_net.testing_size


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
        return result

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